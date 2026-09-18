

SILENT = -1        # item index meaning "no sound at this timestep"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# hip-cat.proj parameter values (single source of truth for defaults)
# ---------------------------------------------------------------------------

HIPCAT = dict(
    # learning rates
    lr_ecin_ca1  = 0.02,  lr_ca1_ecout = 0.002, lr_ecout_ca1 = 0.002,
    lr_ecin_dg   = 0.2,   lr_ecin_ca3  = 0.2,   lr_ca3_ca3   = 0.2,  lr_ca3_ca1 = 0.05,
    # lmix.hebb
    hebb_msp = 0.005, hebb_tsp = 0.05, hebb_ca3_ca1 = 0.005,
    # savg_cor.cor
    savg_cor_msp = 1.0, savg_cor_tsp = 0.4,
    # connectivity
    p_ecin_dg = 0.25, p_ecin_ca3 = 0.25, p_dg_ca3 = 0.05,
    # wt_scale
    abs_ecin_ca1 = 3.0, rel_dg_ca3 = 8.0,
    # wt_sig
    wt_sig_gain = 6.0, wt_sig_off = 1.0, sem_extra = 2.0,
    # cycles
    cycles_auto = 30, cycles_recall = 50, cycles_plus = 20,
    # layer kwta pct
    pct_dg = 0.01, pct_ca3 = 0.06, pct_ca1 = 0.25,
)


# ---------------------------------------------------------------------------
# kWTA inhibition
# ---------------------------------------------------------------------------

def kwta_hard(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Value-preserving k-Winners-Take-All: keep the k largest entries at their
    ORIGINAL values, zero everything else. Used for the clamped EC layers.
    """
    B, N = x.shape
    k = max(1, min(k, N))
    vals, idx = torch.topk(x, k, dim=1)
    out = torch.zeros_like(x)
    out.scatter_(1, idx, vals)
    return out.clamp_(0.0, 1.0)


def kwta(x: torch.Tensor, k: int, gain: float | None = None) -> torch.Tensor:
    """
    Threshold-kWTA. thr = midpoint between k-th and (k+1)-th net input,
    g = relu(net - thr).
      gain=None : g / max(g)                 (v2, scale-free)
      gain=a    : min(g / max(g), a * g)     (v5, drive-aware)
    A layer whose best unit is less than 1/a above threshold stays partially
    active; a well-driven layer gives exactly the v2 result.
    """
    B, N = x.shape
    k = max(1, min(k, N))
    vals, _ = torch.sort(x, dim=1, descending=True)
    if k < N:
        thresh = 0.5 * (vals[:, k - 1] + vals[:, k]).unsqueeze(1)
    else:
        thresh = vals[:, -1].unsqueeze(1) - 1e-6
    g  = (x - thresh).clamp(min=0.0)
    mx = g.max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    a  = g / mx
    if gain is not None and gain > 0:
        a = torch.minimum(a, gain * g)
    return a


# ---------------------------------------------------------------------------
# Weight contrast enhancement (LeabraConSpec wt_sig)
# ---------------------------------------------------------------------------

def sig_wt(w_lin: torch.Tensor, gain: float, off: float) -> torch.Tensor:
    """w_eff = 1 / (1 + (off (1-w)/w)^gain).  gain<=0 -> identity."""
    if gain <= 0:
        return w_lin
    w = w_lin.clamp(1e-6, 1.0 - 1e-6)
    return 1.0 / (1.0 + (off * (1.0 - w) / w) ** gain)


def slay_act_scale(savg: float, lay_sz: int, n_cons: float, sem_extra: float) -> float:
    """
    [approx] Leabra SLayActScale: 1 / expected number of active inputs.
    Full projection: 1 / (# active sending units).
    Partial: 1 / min(max possible, expected + sem_extra * binomial SD).
    """
    act_n = max(1, int(round(savg * lay_sz)))
    if n_cons >= lay_sz:
        return 1.0 / act_n
    r_avg = savg * n_cons
    r_sd  = math.sqrt(max(r_avg * (1.0 - savg), 0.0))
    r_act = min(float(min(int(n_cons), act_n)), r_avg + sem_extra * r_sd)
    return 1.0 / max(r_act, 1.0)


# ---------------------------------------------------------------------------
# Leabra CHL + CPCA weight update  [proj: LeabraConSpec, use_chl=1, err_sb=1]
# ---------------------------------------------------------------------------

@torch.no_grad()
def chl_update(
    W:        torch.Tensor,           # LINEAR weights [out, in], in [0,1]
    mask:     torch.Tensor | None,
    pre_m:    torch.Tensor,
    post_m:   torch.Tensor,
    pre_p:    torch.Tensor,
    post_p:   torch.Tensor,
    lr:       float,
    hebb_mix: float,
    savg:     float = 0.1,            # sending-layer target activity (k/N)
    savg_cor: float = 0.4,            # savg_cor.cor
    err_sb:   bool  = True,           # lmix.err_sb
):
    """
    dW = lr [ hebb_mix * hebb + (1-hebb_mix) * err_sb(err) ]
      err  = <post_p pre_p> - <post_m pre_m>
      hebb = <post_p ( pre_p (m - W) - (1-pre_p) W )>  =  m <post_p pre_p> - W <post_p>
    SUMMED over the batch: each stream is an independent hip-cat trial and the
    learning rates are per trial. Averaging divided every trial by B and, with
    ~5 % of streams inside a word at any pip, starved the lr=0.002 readout.
    Weights are clipped to [0,1] (Leabra wt_limits MIN_MAX).
    """
    B = pre_m.size(0)
    co_p = torch.einsum("bo,bi->oi", post_p, pre_p)      # summed over B streams
    co_m = torch.einsum("bo,bi->oi", post_m, pre_m)

    err = co_p - co_m
    if err_sb:
        err = torch.where(err > 0, err * (1.0 - W), err * W)

    m    = 0.5 / (savg_cor * max(savg, 1e-3) + (1.0 - savg_cor) * 0.5)
    hebb = m * co_p - W * post_p.sum(dim=0).unsqueeze(1)

    dW = lr * (hebb_mix * hebb + (1.0 - hebb_mix) * err)
    if mask is not None:
        dW = dW * mask
    W.add_(dW)
    W.clamp_(0.0, 1.0)
    if mask is not None:
        W.mul_(mask)


# ---------------------------------------------------------------------------
# Connection modules
# ---------------------------------------------------------------------------

class _Conn(nn.Module):
    """Common machinery: wt_sig, abs/rel, act-scale, savg for CPCA."""

    def _init_common(self, wt_abs, wt_rel, learn, lr, hebb_mix, savg_cor,
                     wt_sig_gain, wt_sig_off):
        self.wt_abs      = float(wt_abs)
        self.wt_rel      = float(wt_rel)
        self.learn       = bool(learn)
        self.lr          = float(lr)
        self.hebb_mix    = float(hebb_mix)
        self.savg_cor    = float(savg_cor)
        self.wt_sig_gain = float(wt_sig_gain)
        self.wt_sig_off  = float(wt_sig_off)
        self.act_scale   = 1.0      # SLayActScale, set by the model
        self.savg        = 0.1      # sending layer k/N, set by the model

    def set_sending(self, k_send: int, n_send: int, n_cons: float, sem_extra: float):
        self.savg      = k_send / max(n_send, 1)
        self.act_scale = slay_act_scale(self.savg, n_send, n_cons, sem_extra)

    def eff_weight(self) -> torch.Tensor:
        w = sig_wt(self.weight, self.wt_sig_gain, self.wt_sig_off)
        return w if self.mask is None else w * self.mask

    def forward(self, x):
        return F.linear(x, self.eff_weight()) * self.act_scale

    effective_trials = 8.0   # set by the model; batch-summed dW is scaled by this / B

    def update(self, pre_m, post_m, pre_p, post_p):
        if not self.learn or self.lr == 0:
            return
        B = pre_m.size(0)
        lr = self.lr * min(1.0, self.effective_trials / B)
        chl_update(self.weight, self.mask, pre_m, post_m, pre_p, post_p,
                   lr, self.hebb_mix, savg=self.savg, savg_cor=self.savg_cor)


def _uniform(shape, mean, var, g=None):
    """Leabra UNIFORM rnd: mean +/- var (var = half range)."""
    r = torch.rand(*shape, generator=g) if g is not None else torch.rand(*shape)
    return (r * (2 * var) + (mean - var)).clamp_(0.0, 1.0)


class FullConn(_Conn):
    """Fully-connected learnable projection (FullPrjnSpec)."""

    def __init__(self, in_f, out_f, lr, hebb_mix,
                 wt_abs: float = 1.0, wt_rel: float = 1.0,
                 init_mean: float = 0.5, init_var: float = 0.25,
                 learn: bool = True, no_self: bool = False,
                 savg_cor: float = 0.4,
                 wt_sig_gain: float = HIPCAT["wt_sig_gain"],
                 wt_sig_off:  float = HIPCAT["wt_sig_off"],
                 seed: int | None = None):
        super().__init__()
        self._init_common(wt_abs, wt_rel, learn, lr, hebb_mix, savg_cor,
                          wt_sig_gain, wt_sig_off)
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        W = _uniform((out_f, in_f), init_mean, init_var, g)
        if no_self:                       # FullPrjnSpec self_con=0
            if in_f != out_f:
                raise ValueError("no_self=True requires a square projection")
            mask = 1.0 - torch.eye(out_f)
            W = W * mask
            self.register_buffer("mask", mask)
            self.n_cons = out_f - 1
        else:
            self.mask = None
            self.n_cons = in_f
        self.weight = nn.Parameter(W)


class SparseConn(_Conn):
    """Sparse random projection (UniformRndPrjnSpec), fixed mask, learnable."""

    def __init__(self, in_f, out_f, p_con, lr, hebb_mix,
                 wt_abs: float = 1.0, wt_rel: float = 1.0,
                 init_mean: float = 0.5, init_var: float = 0.25,
                 learn: bool = True, savg_cor: float = 0.4,
                 wt_sig_gain: float = HIPCAT["wt_sig_gain"],
                 wt_sig_off:  float = HIPCAT["wt_sig_off"],
                 seed: int | None = None):
        super().__init__()
        self._init_common(wt_abs, wt_rel, learn, lr, hebb_mix, savg_cor,
                          wt_sig_gain, wt_sig_off)
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        mask = (torch.rand(out_f, in_f, generator=g) < p_con).float()
        dead = mask.sum(1) == 0
        if dead.any():
            cols = torch.randint(in_f, (int(dead.sum().item()),), generator=g)
            mask[dead.nonzero(as_tuple=True)[0], cols] = 1.0
        self.register_buffer("mask", mask)
        self.n_cons = float(p_con * in_f)
        W = _uniform((out_f, in_f), init_mean, init_var, g) * mask
        self.weight = nn.Parameter(W)


class FixedConn(_Conn):
    """Non-learning projection (Mossy fibres, big loop). Weight is a buffer."""

    def __init__(self, in_f, out_f, p_con: float = 1.0,
                 wt_abs: float = 1.0, wt_rel: float = 1.0,
                 init_mean: float = 0.9, init_var: float = 0.01,
                 wt_sig_gain: float = HIPCAT["wt_sig_gain"],
                 wt_sig_off:  float = HIPCAT["wt_sig_off"],
                 seed: int | None = None):
        super().__init__()
        self._init_common(wt_abs, wt_rel, False, 0.0, 0.0, 0.4,
                          wt_sig_gain, wt_sig_off)
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        if p_con < 1.0:
            mask = (torch.rand(out_f, in_f, generator=g) < p_con).float()
            dead = mask.sum(1) == 0
            if dead.any():
                cols = torch.randint(in_f, (int(dead.sum().item()),), generator=g)
                mask[dead.nonzero(as_tuple=True)[0], cols] = 1.0
            self.n_cons = float(p_con * in_f)
        else:
            mask = torch.ones(out_f, in_f)
            self.n_cons = in_f
        W = _uniform((out_f, in_f), init_mean, init_var, g) * mask
        self.register_buffer("weight", W)
        self.register_buffer("mask",   mask)

    def update(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Net-input helper (Leabra abs*rel/sum(rel) normalisation)
# ---------------------------------------------------------------------------

def net_sum(*contribs):
    """
    contribs: (signal_tensor, abs, rel) tuples. signal_tensor already carries
    the act_scale (FullConn/SparseConn/FixedConn.forward applies it).
    Returns sum_p abs_p rel_p signal_p / sum_p rel_p; rel=0 -> projection off
    and removed from the normaliser (Leabra Compute_NetinScale).
    """
    total_w = 0.0
    out     = None
    for sig, a, r in contribs:
        if r == 0 or sig is None:
            continue
        w = a * r
        out = (sig * w) if out is None else (out + sig * w)
        total_w += r
    if out is None or total_w == 0:
        return None
    return out / total_w


# ---------------------------------------------------------------------------
# HipCat model
# ---------------------------------------------------------------------------

class HipCatModel(nn.Module):
    """
    hip-cat on the ABCD-in-noise paradigm.

    Per pip (trial):
      AUTO   minus: EC_in->CA1 on,  CA3->CA1 off          -> act_mid  (MSP minus)
      RECALL minus: EC_in->CA1 off, CA3->CA1 on, CA1 decayed -> act_m (TSP minus)
      PLUS        : both on, EC_out clamped to NEXT item  -> act_p
    Prediction read from EC_out at the end of RECALL (hip-cat act_m).
    """

    # Theta gating [proj]: hard on/off via wt_scale.rel. The keys are
    # multipliers on the projections' base rel so that analysis code can
    # soften them if it wants to. Old names kept for compatibility:
    # TROUGH == auto-encoder minus, PEAK == recall minus.
    THETA_AUTO   = dict(ecin_ca1_rel_mult=1.0, ca3_ca1_rel_mult=0.0,
                        dg_ca3_rel_mult=1.0,  ca3_ca3_rel_mult=1.0, recall_decay=False)
    THETA_RECALL = dict(ecin_ca1_rel_mult=0.0, ca3_ca1_rel_mult=1.0,
                        dg_ca3_rel_mult=0.125, ca3_ca3_rel_mult=1.0, recall_decay=True)
    THETA_PLUS   = dict(ecin_ca1_rel_mult=1.0, ca3_ca1_rel_mult=1.0,
                        dg_ca3_rel_mult=1.0,  ca3_ca3_rel_mult=1.0, recall_decay=False)
    THETA_TROUGH = THETA_AUTO
    THETA_PEAK   = THETA_RECALL


    def __init__(
        self,
        input_size:     int,
        # layer sizes [proj]
        n_dg:           int = 400,
        n_ca3:          int = 80,
        n_ca1:          int = 100,
        # kWTA k values [proj pct * n], EC layers [paradigm]
        k_ecin:         int = 2,      # two-hot window (cur, prev)
        k_ecout:        int = 1,      # one-hot next-item target
        k_dg:           int | None = None,    # pct 0.01 -> 4
        k_ca3:          int | None = None,    # pct 0.06 -> 5
        k_ca1:          int | None = None,    # pct 0.25 -> 25
        # connectivity [proj]
        p_ecin_dg:      float = HIPCAT["p_ecin_dg"],
        p_ecin_ca3:     float = HIPCAT["p_ecin_ca3"],
        p_dg_ca3:       float = HIPCAT["p_dg_ca3"],
        # MSP learning rates [proj]
        lr_ecin_ca1:    float = HIPCAT["lr_ecin_ca1"],
        lr_ca1_ecout:   float = HIPCAT["lr_ca1_ecout"],
        lr_ecout_ca1:   float = HIPCAT["lr_ecout_ca1"],
        # TSP learning rates. .proj: 0.2/0.2/0.2/0.05. With per-stream updates
        # summed over the batch, 0.2 made the readout swing epoch to epoch;
        # 0.05 for the three encoding projections converges (5 epochs, B=32,
        # T=200: TSP readout B->C 1.00, C->D 1.00). CA3->CA1 keeps 0.05.
        lr_ecin_dg:     float = 0.05,   # .proj 0.2; see note above
        lr_ecin_ca3:    float = 0.05,
        lr_ca3_ca3:     float = 0.05,
        lr_ca3_ca1:     float = HIPCAT["lr_ca3_ca1"],
        # Hebbian mixes [proj]
        hebb_msp:       float = HIPCAT["hebb_msp"],
        hebb_tsp:       float = HIPCAT["hebb_tsp"],
        hebb_ca3_ca1:   float = HIPCAT["hebb_ca3_ca1"],
        # CA3->CA3: CA3 is identical in the recall and plus phases (nothing feeds
        # back into CA3), so the CHL error term is zero for this projection and
        # hip-cat's setting (lr 0.2, hebb 0.05) is de facto Hebbian at 0.01 per
        # trial. Here the Hebbian (CPCA) mix is set to 1 so the attractor
        # actually forms; with hebb 0.05 the CPCA decay term wins and the
        # effective recurrent weights stay at ~0.                         [v6]
        hebb_ca3_ca3:   float = HIPCAT["hebb_tsp"],
        # wt_sig contrast enhancement [proj]; gain=0 -> raw weights (v3)
        wt_sig_gain:    float = HIPCAT["wt_sig_gain"],
        wt_sig_off:     float = HIPCAT["wt_sig_off"],
        # per-projection act scaling [proj, approx]; False -> v2 behaviour
        act_scale:      bool  = True,
        # drive-aware kWTA gain [v5]; None -> v2 scale-free kWTA
        act_gain:       float | None = None,
        # Batch = B independent streams = B hip-cat trials. Per-trial updates are
        # summed, then scaled by effective_trials/B: one step of B summed
        # error-driven updates overshoots what B sequential trials would do
        # (sequential updates self-limit as the error shrinks). 8 was found
        # empirically at B=32: x1 flips the readout between epochs, x0.25 is
        # stable. [v5]
        effective_trials: float = 8.0,
        # cycles [proj]; settle_steps is a legacy override for all three
        cycles_auto:    int | None = None,
        cycles_recall:  int | None = None,
        cycles_plus:    int | None = None,
        settle_steps:   int | None = None,
        recall_decay:   bool  = True,    # CA1LayerSpec recall_decay=1
        # Decay CA3 as well at recall onset [v6]: the recall-phase CA3 is then
        # rebuilt from the cue by settling (perforant + weak mossy + recurrence)
        # rather than inherited from the mossy-dominated auto phase. With the
        # weak mossy input the first cycles mis-select among the mossy
        # candidates, so recall != plus and CHL trains CA3->CA3 to repair the
        # selection - the same computation that completes a degraded cue.
        recall_decay_ca3: bool = True,
        initial_cycles: int   = 1,       # cycle of the "initial" eval snapshot
        # phase the PREDICTION is read from: "auto" = MSP (as in the dissertation)
        eval_readout:   str   = "auto",
        snapshot_phase: str   = "recall",      # phase whose layer states are snap["settled"] (hip-cat act_m)
        # temporal asymmetry of the two-hot window [paradigm]
        prev_activity:  float = 0.9,
        # How the previous tone is carried into EC_in [v6]:
        #   "separate": EC_in has 2N units, N for the current tone and N for the
        #               previous tone. "previous = B" then occurs in exactly one
        #               stored pattern ([C,B]), so at a gap the partial cue
        #               selects [C,B] and auto-associative CA3->CA3 completes it.
        #   "shared":   the dissertation's two-hot code on N units (1.0 / 0.9).
        #               "B before" and "B now" are the same unit, so an omission
        #               is an ambiguous cue and only the big loop can fill it.
        slot_code:      str   = "separate",
        # decay.event: state_carry = 1 - decay.event. hip-cat: decay.event=1.
        state_carry:    float = 0.0,
        # sequence completion [v5]
        predictive_ca3: bool | None = None,    # True = heteroassociative CA3->CA3; False = auto-associative (hip-cat)
        mossy_recall_mult: float = 0.125,      # mossy rel in RECALL = 8 * this
        rec_recall_mult:   float = 2.0,        # CA3->CA3 rel in RECALL during TRAINING
        # CA3->CA3 rel in RECALL at TEST (eval_step). Stronger recurrent gain at
        # retrieval than during learning: with 2 the gap CA3 matches the
        # standard C code at 0.52, with 4 at 1.00 (CA3->CA3 lesioned: 0.52).
        # Trained at 4 throughout the split was 0.77 vs 0.58 (one seed).   [v6]
        rec_recall_mult_eval: float = 4.0,
        ecout_ca1_recall_mult: float = 0.25,   # EC_out->CA1 rel in RECALL = 1 * this
        # EC_in->CA3 (perforant path) rel in RECALL = 1 * this. Kept ON: at a
        # gap the perforant cue is the thing that is degraded, so it must reach
        # CA3 for the recurrent collaterals to have something to repair.   [v6]
        perforant_recall_mult: float = 1.0,
        theta_gates_ca3: bool | None = None,   # legacy alias: False -> mossy_recall_mult=1
        # big loop from the previous pip's EC_out (its prediction for this pip)
        bigloop_from_prev: bool = True,
        # gain on the recalled pattern entering EC_in. hip-cat: 1.0
        # (abs 2 * rel 0.5 * w 0.5 == Input's 1 * 1 * 0.5). 0.8 keeps a present
        # item above a wrong recall in the k=2 competition.
        bigloop_gain:   float | None = None,
        device:         str | None = None,
        seed:           int | None = None,
    ):
        super().__init__()
        self.input_size     = int(input_size)
        self.n_dg           = int(n_dg)
        self.n_ca3          = int(n_ca3)
        self.n_ca1          = int(n_ca1)
        self.k_ecin         = int(k_ecin)
        self.k_ecout        = int(k_ecout)
        self.k_dg  = int(k_dg)  if k_dg  is not None else max(1, int(round(HIPCAT["pct_dg"]  * n_dg)))
        self.k_ca3 = int(k_ca3) if k_ca3 is not None else max(1, int(round(HIPCAT["pct_ca3"] * n_ca3)))
        self.k_ca1 = int(k_ca1) if k_ca1 is not None else max(1, int(round(HIPCAT["pct_ca1"] * n_ca1)))
        if settle_steps is not None:
            cycles_auto   = cycles_auto   if cycles_auto   is not None else settle_steps
            cycles_recall = cycles_recall if cycles_recall is not None else settle_steps
            cycles_plus   = cycles_plus   if cycles_plus   is not None else settle_steps
        self.cycles_auto    = int(cycles_auto   if cycles_auto   is not None else HIPCAT["cycles_auto"])
        self.cycles_recall  = int(cycles_recall if cycles_recall is not None else HIPCAT["cycles_recall"])
        self.cycles_plus    = int(cycles_plus   if cycles_plus   is not None else HIPCAT["cycles_plus"])
        self.settle_steps   = self.cycles_auto      # legacy attribute
        self.recall_decay   = bool(recall_decay)
        self.recall_decay_ca3 = bool(recall_decay_ca3)
        self.initial_cycles = int(initial_cycles)
        if eval_readout not in ("recall", "auto"):
            raise ValueError("eval_readout must be 'recall' or 'auto'")
        self.eval_readout   = eval_readout
        if snapshot_phase not in ("recall", "auto"):
            raise ValueError("snapshot_phase must be 'recall' or 'auto'")
        self.snapshot_phase = snapshot_phase
        self.prev_activity  = float(prev_activity)
        if slot_code not in ("separate", "shared"):
            raise ValueError("slot_code must be 'separate' or 'shared'")
        self.slot_code = slot_code
        self.n_ecin    = 2 * self.input_size if slot_code == "separate" else self.input_size
        if bigloop_gain is None:
            bigloop_gain = 0.0 if slot_code == "separate" else 0.8
        self.state_carry    = float(state_carry)
        self.act_gain       = act_gain
        self.predictive_ca3 = bool(predictive_ca3)
        if theta_gates_ca3 is False:
            mossy_recall_mult = 1.0
        self.mossy_recall_mult = float(mossy_recall_mult)
        self.rec_recall_mult   = float(rec_recall_mult)
        self.rec_recall_mult_eval = float(rec_recall_mult_eval)
        self.ecout_ca1_recall_mult = float(ecout_ca1_recall_mult)
        self.perforant_recall_mult = float(perforant_recall_mult)
        self.THETA_RECALL = dict(HipCatModel.THETA_RECALL, dg_ca3_rel_mult=self.mossy_recall_mult,
                                 ca3_ca3_rel_mult=self.rec_recall_mult,
                                 ecout_ca1_rel_mult=self.ecout_ca1_recall_mult,
                                 ecin_ca3_rel_mult=self.perforant_recall_mult)
        self.THETA_PEAK   = self.THETA_RECALL
        self.bigloop_from_prev = bool(bigloop_from_prev)
        self.bigloop_gain   = float(bigloop_gain)
        self.use_act_scale  = bool(act_scale)
        self.effective_trials = float(effective_trials)
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # stream state
        self._carry         = None   # (DG, CA3, CA1, ECout) final states of previous pip
        self._prev_ecout    = None   # big-loop source (previous pip's final EC_out)
        self._prev_cur_slot = None   # delay line: previous pip's current-slot one-hot
        self._prev_ca3      = None   # recurrent source: previous pip's final CA3

        # ── Pathway switches (per-projection lesion control) ─────────────────
        self.path_on = {
            "ecin_dg":   True,
            "ecin_ca3":  True,
            "dg_ca3":    True,
            "ca3_ca3":   True,
            "ca3_ca1":   True,
            "ecin_ca1":  True,
            "ecout_ca1": True,
            "ca1_ecout": True,
            "bigloop":   True,   # EC_out -> EC_in feedback
        }

        sd = (lambda i: None) if seed is None else (lambda i: seed + i)
        sig = dict(wt_sig_gain=wt_sig_gain, wt_sig_off=wt_sig_off)
        msp_cor, tsp_cor = HIPCAT["savg_cor_msp"], HIPCAT["savg_cor_tsp"]

        # ── MSP connections (HippoEncoderConSpec) ────────────────────────────
        n_ecin = self.n_ecin
        self.ecin_to_ca1  = FullConn(n_ecin, n_ca1, lr_ecin_ca1, hebb_msp,
                                     wt_abs=HIPCAT["abs_ecin_ca1"], wt_rel=1.0,
                                     savg_cor=msp_cor, seed=sd(10), **sig)
        self.ca1_to_ecout = FullConn(n_ca1, input_size, lr_ca1_ecout, hebb_msp,
                                     wt_abs=1.0, wt_rel=1.0,
                                     savg_cor=msp_cor, seed=sd(11), **sig)
        self.ecout_to_ca1 = FullConn(input_size, n_ca1, lr_ecout_ca1, hebb_msp,
                                     wt_abs=1.0, wt_rel=1.0,
                                     savg_cor=msp_cor, seed=sd(12), **sig)

        # ── TSP connections (XCalCHLConSpec, use_chl=1) ──────────────────────
        self.ecin_to_dg  = SparseConn(n_ecin, n_dg,  p_ecin_dg,  lr_ecin_dg,  hebb_tsp,
                                      savg_cor=tsp_cor, seed=sd(1), **sig)
        self.ecin_to_ca3 = SparseConn(n_ecin, n_ca3, p_ecin_ca3, lr_ecin_ca3, hebb_tsp,
                                      savg_cor=tsp_cor, seed=sd(2), **sig)
        # Mossy: fixed, w=0.9+-0.01, rel=8, never gated by theta
        self.dg_to_ca3   = FixedConn(n_dg, n_ca3, p_dg_ca3,
                                     wt_abs=1.0, wt_rel=HIPCAT["rel_dg_ca3"],
                                     init_mean=0.9, init_var=0.01, seed=sd(3), **sig)
        # Recurrent collaterals: full, no autapses (self_con=0)
        self.ca3_to_ca3  = FullConn(n_ca3, n_ca3, lr_ca3_ca3, hebb_ca3_ca3,
                                    no_self=True, savg_cor=tsp_cor, seed=sd(5), **sig)
        # Schaffer collaterals: TSP lrate, MSP-style hebb 0.005
        self.ca3_to_ca1  = FullConn(n_ca3, n_ca1, lr_ca3_ca1, hebb_ca3_ca1,
                                    savg_cor=tsp_cor, seed=sd(6), **sig)

        # ── Big loop EC_out -> EC_in: one-to-one, fixed ──────────────────────
        self.ecout_to_ecin = FixedConn(input_size, n_ecin, p_con=1.0,
                                       wt_abs=2.0, wt_rel=0.5,
                                       init_mean=1.0, init_var=0.0,
                                       wt_sig_gain=0.0, seed=sd(4))
        eye = torch.zeros(n_ecin, input_size)          # EC_out -> current-slot units
        eye[:input_size, :] = torch.eye(input_size)
        self.ecout_to_ecin.weight.copy_(eye)
        self.ecout_to_ecin.mask.copy_(eye)

        # ── Sending-layer activity scaling (SLayActScale) ────────────────────
        N = input_size
        for conn, k_s, n_s in [
            (self.ecin_to_ca1,  self.k_ecin,  n_ecin),
            (self.ecout_to_ca1, self.k_ecout, N),
            (self.ca1_to_ecout, self.k_ca1,   n_ca1),
            (self.ecin_to_dg,   self.k_ecin,  N),
            (self.ecin_to_ca3,  self.k_ecin,  N),
            (self.dg_to_ca3,    self.k_dg,    n_dg),
            (self.ca3_to_ca3,   self.k_ca3,   n_ca3),
            (self.ca3_to_ca1,   self.k_ca3,   n_ca3),
        ]:
            conn.set_sending(k_s, n_s, conn.n_cons, HIPCAT["sem_extra"])
            if not self.use_act_scale:
                conn.act_scale = 1.0
        self.ecout_to_ecin.act_scale = 1.0
        for conn in [self.ecin_to_ca1, self.ca1_to_ecout, self.ecout_to_ca1, self.ecin_to_dg,
                     self.ecin_to_ca3, self.ca3_to_ca3, self.ca3_to_ca1]:
            conn.effective_trials = self.effective_trials

    # -----------------------------------------------------------------------
    # Lesion control
    # -----------------------------------------------------------------------

    def set_lesion(self, msp: bool = False, tsp: bool = False):
        """Whole-pathway shortcuts. msp=True kills the MSP *learning*
        projections but keeps CA1->EC_out alive so readout still works."""
        self.path_on["ecin_ca1"]  = not msp
        self.path_on["ecout_ca1"] = not msp
        self.path_on["ca1_ecout"] = True
        self.path_on["ecin_dg"]   = not tsp
        self.path_on["ecin_ca3"]  = not tsp
        self.path_on["dg_ca3"]    = not tsp
        self.path_on["ca3_ca3"]   = not tsp
        self.path_on["ca3_ca1"]   = not tsp

    def set_pathway(self, name: str, on: bool):
        if name not in self.path_on:
            raise KeyError(f"unknown pathway {name!r}, valid: {list(self.path_on)}")
        self.path_on[name] = bool(on)

    def reset_lesions(self):
        for k in self.path_on:
            self.path_on[k] = True

    # -----------------------------------------------------------------------
    # Stream state
    # -----------------------------------------------------------------------

    def reset_state(self):
        """Call at the start of every stream (training loop / run_stream do)."""
        self._carry         = None
        self._prev_ecout    = None
        self._prev_cur_slot = None
        self._prev_ca3      = None

    def zero_states(self, B: int):
        dev = self.device
        return (torch.zeros(B, self.n_dg,        device=dev),
                torch.zeros(B, self.n_ca3,       device=dev),
                torch.zeros(B, self.n_ca1,       device=dev),
                torch.zeros(B, self.input_size,  device=dev))

    def _init_states(self, B: int):
        """decay.event: states start at (1 - decay) * previous final states."""
        if self.state_carry <= 0.0 or self._carry is None:
            return self.zero_states(B)
        DG, CA3, CA1, ECout = self._carry
        if CA3.size(0) != B:
            return self.zero_states(B)
        c = self.state_carry
        return DG * c, CA3 * c, CA1 * c, ECout * c

    def _store_states(self, DG, CA3, CA1, ECout):
        self._carry      = (DG.detach(), CA3.detach(), CA1.detach(), ECout.detach())
        self._prev_ecout = ECout.detach()
        self._prev_ca3   = CA3.detach()

    def _prev_or_zeros(self, x, B, n):
        if x is None or x.size(0) != B:
            return torch.zeros(B, n, device=self.device)
        return x

    def _bigloop_source(self, B: int):
        if not self.bigloop_from_prev:
            return None
        if self._prev_ecout is None or self._prev_ecout.size(0) != B:
            return torch.zeros(B, self.input_size, device=self.device)
        return self._prev_ecout

    # -----------------------------------------------------------------------
    # Input construction
    # -----------------------------------------------------------------------

    def make_ecin_input(self, cur_idx: torch.Tensor, prev_idx: torch.Tensor):
        """
        EC_in clamp. current = 1.0, previous = prev_activity.
          slot_code "separate": current tone on units [0, N), previous on [N, 2N)
          slot_code "shared"  : both on the same N units (two-hot)
        A SILENT slot is empty; a SILENT previous slot is filled from the delay
        line (the previous pip's current-slot content, if any).
        """
        B = cur_idx.size(0)
        N = self.input_size
        off = N if self.slot_code == "separate" else 0
        x = torch.zeros(B, self.n_ecin, device=self.device)
        idx_b = torch.arange(B, device=self.device)
        prev_idx, cur_idx = prev_idx.long(), cur_idx.long()
        ok_p, ok_c = prev_idx >= 0, cur_idx >= 0
        if ok_p.any():
            x[idx_b[ok_p], prev_idx[ok_p] + off] = self.prev_activity
        if (~ok_p).any() and self._prev_cur_slot is not None and self._prev_cur_slot.size(0) == B:
            x[~ok_p, off:off + N] = torch.maximum(x[~ok_p, off:off + N],
                                                  self.prev_activity * self._prev_cur_slot[~ok_p])
        if ok_c.any():
            x[idx_b[ok_c], cur_idx[ok_c]] = 1.0
        return x

    def make_target(self, next_idx: torch.Tensor):
        """One-hot plus-phase target at EC_out. SILENT -> all zeros."""
        B = next_idx.size(0)
        t = torch.zeros(B, self.input_size, device=self.device)
        next_idx = next_idx.long()
        ok = next_idx >= 0
        if ok.any():
            idx_b = torch.arange(B, device=self.device)
            t[idx_b[ok], next_idx[ok]] = 1.0
        return t

    def _update_delay_line(self, cur_idx: torch.Tensor, ecin: torch.Tensor,
                           input_clamp: torch.Tensor):
        """Record this pip's current-slot content for the next pip's prev slot."""
        B = cur_idx.size(0)
        cur_idx = cur_idx.long()
        slot = torch.zeros(B, self.input_size, device=self.device)
        ok_c = cur_idx >= 0
        if ok_c.any():
            slot[torch.arange(B, device=self.device)[ok_c], cur_idx[ok_c]] = 1.0
        gap = ~ok_c
        if gap.any():
            # recalled content = EC_in activity on units that carried no clamp
            free = (input_clamp[gap] <= 0).float()
            rec  = ecin[gap] * free
            val, idx = rec.max(dim=1)
            has = val > 0
            if has.any():
                rows = torch.arange(int(gap.sum().item()), device=self.device)[has]
                s = slot[gap]
                s[rows, idx[has]] = 1.0
                slot[gap] = s
        self._prev_cur_slot = slot

    def ecin_activation(self, input_clamp: torch.Tensor, ECout: torch.Tensor):
        """
        EC_in = clamp, plus the big-loop (EC_out -> EC_in, one-to-one) recall on
        units that carry no clamp, then value-preserving top-k.
        """
        x = input_clamp
        if self.path_on["bigloop"] and ECout is not None:
            fb   = self.ecout_to_ecin(ECout)          # identity -> EC_out itself
            free = (input_clamp <= 0).float()
            x = torch.maximum(x, self.bigloop_gain * fb * free)
        return kwta_hard(x, self.k_ecin)

    # -----------------------------------------------------------------------
    # Single settling phase
    # -----------------------------------------------------------------------

    def _theta_rels(self, theta: dict, phase: str):
        ecin_ca1_rel = self.ecin_to_ca1.wt_rel * theta.get("ecin_ca1_rel_mult", 1.0)
        ca3_ca1_rel  = self.ca3_to_ca1.wt_rel  * theta.get("ca3_ca1_rel_mult",  1.0)
        dgm = theta.get("dg_ca3_rel_mult", 1.0)
        ccm = theta.get("ca3_ca3_rel_mult", 1.0)
        return (ecin_ca1_rel, ca3_ca1_rel,
                self.dg_to_ca3.wt_rel * dgm, self.ca3_to_ca3.wt_rel * ccm)

    @torch.no_grad()
    def _settle(
        self,
        input_clamp: torch.Tensor,
        DG_init: torch.Tensor,
        CA3_init: torch.Tensor,
        CA1_init: torch.Tensor,
        ECout_init: torch.Tensor,
        ecout_clamp: torch.Tensor | None,
        theta: dict,
        capture_at: int | None = None,
        capture_all: bool = False,
        n_cycles: int | None = None,
        bigloop_src: torch.Tensor | None = None,
        rec_src: torch.Tensor | None = None,
        phase: str = "auto",
    ):
        """
        Runs one phase of settling.
          snap["pre"]     : state after EC_in clamp, before any update
          snap["initial"] : state after cycle `capture_at` (optional)
          snap["settled"] : final state
        """
        if n_cycles is None:
            n_cycles = {"auto": self.cycles_auto, "recall": self.cycles_recall,
                        "plus": self.cycles_plus}.get(phase, self.cycles_auto)
        DG, CA3, CA1, ECout = (DG_init.clone(), CA3_init.clone(),
                               CA1_init.clone(), ECout_init.clone())
        if theta.get("recall_decay", False) and self.recall_decay:
            # CA1LayerSpec recall_decay. EC_out is decayed with it: with
            # instantaneous units the EC_out<->CA1 loop would otherwise
            # re-instate the auto-phase CA1 state before CA3 can drive it.
            CA1 = torch.zeros_like(CA1)
            if ecout_clamp is None:
                ECout = torch.zeros_like(ECout)
            if self.recall_decay_ca3:
                CA3 = torch.zeros_like(CA3)

        snap = {}
        all_cycles = [] if capture_all else None

        # heteroassociative recurrence [v5]: drive from the PREVIOUS pip's final
        # CA3, held fixed across this pip. predictive_ca3=False -> hip-cat
        # auto-associative settling on the evolving CA3.
        if self.predictive_ca3:
            rec_src   = self._prev_or_zeros(rec_src, input_clamp.size(0), self.n_ca3)
            rec_drive = self.ca3_to_ca3(rec_src)
        else:
            rec_drive = None
        gain = self.act_gain

        # big-loop source: previous pip's EC_out (bigloop_from_prev) or the
        # EC_out evolving in this phase
        if bigloop_src is None and self.bigloop_from_prev:
            bigloop_src = ECout_init
        fb_src = bigloop_src

        ecin_ca1_rel, ca3_ca1_rel, dg_ca3_rel, ca3_ca3_rel = self._theta_rels(theta, phase)
        ecout_ca1_rel = self.ecout_to_ca1.wt_rel * theta.get("ecout_ca1_rel_mult", 1.0)
        ecin_ca3_rel  = self.ecin_to_ca3.wt_rel  * theta.get("ecin_ca3_rel_mult", 1.0)

        ecin0 = self.ecin_activation(input_clamp, fb_src if fb_src is not None else ECout)
        snap["pre"] = {"DG": DG.clone(), "CA3": CA3.clone(), "CA1": CA1.clone(),
                       "ECout": ECout.clone(), "ECin": ecin0.clone()}

        ecin = ecin0
        for cyc in range(1, n_cycles + 1):
            ecin = self.ecin_activation(input_clamp, fb_src if fb_src is not None else ECout)

            # DG
            if self.path_on["ecin_dg"]:
                DG = kwta(self.ecin_to_dg(ecin) * self.ecin_to_dg.wt_abs, self.k_dg, gain)
            else:
                DG = torch.zeros_like(DG)

            # CA3
            ca3_contribs = []
            if self.path_on["ecin_ca3"]:
                ca3_contribs.append((self.ecin_to_ca3(ecin),
                                     self.ecin_to_ca3.wt_abs, ecin_ca3_rel))
            if self.path_on["dg_ca3"]:
                ca3_contribs.append((self.dg_to_ca3(DG), self.dg_to_ca3.wt_abs, dg_ca3_rel))
            if self.path_on["ca3_ca3"]:
                ca3_contribs.append((rec_drive if rec_drive is not None else self.ca3_to_ca3(CA3),
                                     self.ca3_to_ca3.wt_abs, ca3_ca3_rel))
            ca3_net = net_sum(*ca3_contribs)
            CA3 = kwta(ca3_net, self.k_ca3, gain) if ca3_net is not None else torch.zeros_like(CA3)

            # CA1
            ca1_contribs = []
            if self.path_on["ecin_ca1"]:
                ca1_contribs.append((self.ecin_to_ca1(ecin), self.ecin_to_ca1.wt_abs, ecin_ca1_rel))
            if self.path_on["ca3_ca1"]:
                ca1_contribs.append((self.ca3_to_ca1(CA3), self.ca3_to_ca1.wt_abs, ca3_ca1_rel))
            if self.path_on["ecout_ca1"]:
                ca1_contribs.append((self.ecout_to_ca1(ECout),
                                     self.ecout_to_ca1.wt_abs, ecout_ca1_rel))
            ca1_net = net_sum(*ca1_contribs)
            CA1 = kwta(ca1_net, self.k_ca1, gain) if ca1_net is not None else torch.zeros_like(CA1)

            # EC_out
            if ecout_clamp is not None:
                ECout = kwta_hard(ecout_clamp, self.k_ecout)
            elif self.path_on["ca1_ecout"]:
                ECout = kwta(self.ca1_to_ecout(CA1), self.k_ecout, gain)

            if capture_at is not None and cyc == capture_at:
                snap["initial"] = {"DG": DG.clone(), "CA3": CA3.clone(), "CA1": CA1.clone(),
                                   "ECout": ECout.clone(), "ECin": ecin.clone()}
            if capture_all:
                all_cycles.append({"cyc": cyc,
                                   "DG": DG[0].clone().cpu(), "CA3": CA3[0].clone().cpu(),
                                   "CA1": CA1[0].clone().cpu(), "ECout": ECout[0].clone().cpu(),
                                   "ECin": ecin[0].clone().cpu()})

        if capture_all:
            snap["all_cycles"] = all_cycles
        snap["settled"] = {"DG": DG, "CA3": CA3, "CA1": CA1, "ECout": ECout, "ECin": ecin}
        return DG, CA3, CA1, ECout, snap

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def train_step(
        self,
        cur_idx:  torch.Tensor,    # [B] long  current item
        prev_idx: torch.Tensor,    # [B] long  previous item
        next_idx: torch.Tensor,    # [B] long  next item (target)
    ):
        """
        One hip-cat trial: AUTO minus -> RECALL minus -> PLUS, then CHL.

        Returns:
          pred_err: [B] MSE of the RECALL-phase EC_out vs target (hip-cat act_m)
          pred_idx: [B] argmax of the RECALL-phase EC_out
          aux: snapshot dict (also holds the auto-phase prediction as pred_idx_auto)
        """
        B = cur_idx.size(0)
        DG, CA3, CA1, ECout = self._init_states(B)
        bl  = self._bigloop_source(B)
        rec = self._prev_ca3
        input_clamp = self.make_ecin_input(cur_idx, prev_idx)
        target_oh   = self.make_target(next_idx)

        # ── AUTO-ENCODER minus (act_mid) ─────────────────────────────────────
        DG_m1, CA3_m1, CA1_m1, ECout_m1, snap_m1 = self._settle(
            input_clamp, DG, CA3, CA1, ECout, ecout_clamp=None,
            theta=self.THETA_AUTO, bigloop_src=bl, rec_src=rec, phase="auto")
        ecin_m1 = snap_m1["settled"]["ECin"]

        # ── RECALL minus (act_m) ─────────────────────────────────────────────
        DG_m2, CA3_m2, CA1_m2, ECout_m2, snap_m2 = self._settle(
            input_clamp, DG_m1, CA3_m1, CA1_m1, ECout_m1, ecout_clamp=None,
            theta=self.THETA_RECALL, bigloop_src=bl, rec_src=rec, phase="recall")
        ecin_m2 = snap_m2["settled"]["ECin"]

        # ── PLUS (act_p): EC_out clamped to the next item ────────────────────
        DG_p, CA3_p, CA1_p, ECout_p, snap_p = self._settle(
            input_clamp, DG_m2, CA3_m2, CA1_m2, ECout_m2, ecout_clamp=target_oh,
            theta=self.THETA_PLUS, bigloop_src=bl, rec_src=rec, phase="plus")
        ecin_p = snap_p["settled"]["ECin"]

        # ── Weight updates ───────────────────────────────────────────────────
        # MSP (HippoEncoderConSpec): act_mid vs act_p
        if self.path_on["ecin_ca1"]:
            self.ecin_to_ca1.update(ecin_m1, CA1_m1, ecin_p, CA1_p)
        if self.path_on["ca1_ecout"]:
            self.ca1_to_ecout.update(CA1_m1, ECout_m1, CA1_p, ECout_p)
        if self.path_on["ecout_ca1"]:
            self.ecout_to_ca1.update(ECout_m1, CA1_m1, ECout_p, CA1_p)
        # TSP (XCalCHLConSpec): act_m vs act_p
        if self.path_on["ecin_dg"]:
            self.ecin_to_dg.update(ecin_m2, DG_m2, ecin_p, DG_p)
        if self.path_on["ecin_ca3"]:
            self.ecin_to_ca3.update(ecin_m2, CA3_m2, ecin_p, CA3_p)
        if self.path_on["ca3_ca3"]:
            if self.predictive_ca3:
                # pre = previous pip's CA3 (same in both phases): the recurrent
                # weights learn to retrieve this pip's PLUS code from it.
                if rec is not None and rec.size(0) == B:
                    self.ca3_to_ca3.update(rec, CA3_m2, rec, CA3_p)
            else:
                self.ca3_to_ca3.update(CA3_m2, CA3_m2, CA3_p, CA3_p)   # hip-cat
        if self.path_on["ca3_ca1"]:
            self.ca3_to_ca1.update(CA3_m2, CA1_m2, CA3_p, CA1_p)
        # dg_to_ca3, ecout_to_ecin: fixed

        # ── Prediction from the RECALL phase (hip-cat act_m) ─────────────────
        target_kw = kwta_hard(target_oh, self.k_ecout)
        pred_err  = ((ECout_m2 - target_kw) ** 2).mean(dim=1)
        pred_idx  = ECout_m2.argmax(dim=1)

        # end-of-trial state (plus phase) is what the next pip sees
        self._store_states(DG_p, CA3_p, CA1_p, ECout_p)
        self._update_delay_line(cur_idx, ecin_p, input_clamp)

        return pred_err, pred_idx, dict(
            DG_m1=DG_m1, CA3_m1=CA3_m1, CA1_m1=CA1_m1, ECout_m1=ECout_m1,
            DG_m2=DG_m2, CA3_m2=CA3_m2, CA1_m2=CA1_m2, ECout_m2=ECout_m2,
            DG_p=DG_p,   CA3_p=CA3_p,   CA1_p=CA1_p,   ECout_p=ECout_p,
            pred_idx_auto=ECout_m1.argmax(dim=1),
            pred_err_auto=((ECout_m1 - target_kw) ** 2).mean(dim=1),
        )

    # -----------------------------------------------------------------------
    # Eval step (no learning)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def eval_step(
        self,
        cur_idx:  torch.Tensor,
        prev_idx: torch.Tensor,
        next_idx: torch.Tensor,
        capture_initial: bool = True,
        capture_all:     bool = False,
    ):
        """
        Runs AUTO then RECALL minus phases (no plus, no learning).
        Prediction (and the big-loop feedback to the next pip) is read from the
        eval_readout phase (default "auto" = MSP); layer states in snap["settled"]
        come from snapshot_phase (default "recall" = end of the full settle).

        snap keys: "pre", "initial" (cycle initial_cycles of the AUTO phase,
        if capture_initial), "auto" (end of AUTO), "recall" (end of RECALL),
        "settled" (= the readout phase's final state).
        """
        B = cur_idx.size(0)
        DG, CA3, CA1, ECout = self._init_states(B)
        bl  = self._bigloop_source(B)
        rec = self._prev_ca3
        input_clamp = self.make_ecin_input(cur_idx, prev_idx)
        target_oh   = self.make_target(next_idx)

        DG_a, CA3_a, CA1_a, ECout_a, snap_a = self._settle(
            input_clamp, DG, CA3, CA1, ECout, ecout_clamp=None,
            theta=self.THETA_AUTO,
            capture_at=self.initial_cycles if capture_initial else None,
            capture_all=capture_all, bigloop_src=bl, rec_src=rec, phase="auto")
        theta_recall_eval = dict(self.THETA_RECALL, ca3_ca3_rel_mult=self.rec_recall_mult_eval)
        DG_r, CA3_r, CA1_r, ECout_r, snap_r = self._settle(
            input_clamp, DG_a, CA3_a, CA1_a, ECout_a, ecout_clamp=None,
            theta=theta_recall_eval, capture_all=capture_all,
            bigloop_src=bl, rec_src=rec, phase="recall")

        snap = {"pre": snap_a["pre"], "auto": snap_a["settled"], "recall": snap_r["settled"]}
        if "initial" in snap_a:
            snap["initial"] = snap_a["initial"]
        if capture_all:
            snap["all_cycles"] = snap_a["all_cycles"] + [
                dict(c, cyc=c["cyc"] + self.cycles_auto) for c in snap_r["all_cycles"]]

        ECout_s = ECout_r if self.eval_readout == "recall" else ECout_a
        snap["settled"] = snap_r["settled"] if self.snapshot_phase == "recall" else snap_a["settled"]

        target_kw = kwta_hard(target_oh, self.k_ecout)
        pred_err  = ((ECout_s - target_kw) ** 2).mean(dim=1)
        pred_idx  = ECout_s.argmax(dim=1)

        # carried to the next pip: layer states from the snapshot phase, big-loop
        # source = the readout phase's EC_out (its prediction for the next pip)
        DG_c, CA3_c, CA1_c = ((DG_r, CA3_r, CA1_r) if self.snapshot_phase == "recall"
                              else (DG_a, CA3_a, CA1_a))
        self._store_states(DG_c, CA3_c, CA1_c, ECout_s)
        self._update_delay_line(cur_idx, snap["settled"]["ECin"], input_clamp)
        return pred_err, pred_idx, snap
