"""
  pred_err_all          mean MSE forward-pred error over all timesteps
  pred_err_word         mean over within-word transitions (B/C/D positions)
  pred_err_bg           mean over background transitions
  acc_all               argmax accuracy on all timesteps
  acc_AB, acc_BC, acc_CD, acc_Dbg, acc_bg   per-transition accuracy
  hist_per_pos          per-epoch dict with all of the above

Key result to look for:
  acc_BC and acc_CD should rise above chance (1/16) and approach 1.0
  faster than acc_bg (which should stay at chance).
"""

import argparse
import os
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import HipCatModel, HIPCAT
from data  import make_embedded_stream, sound_size, WORD, ALPHABET_SIZE


# ── Helpers ───────────────────────────────────────────────────────────────────

def epoch_metrics(pred_err_t, pred_acc_t, word_pos):
    """
    pred_err_t, pred_acc_t : list of [B] tensors over T timesteps
    word_pos               : [B, T] long  (-1 bg; 0..3 within word)

    Returns dict of per-transition means.
    """
    err = torch.stack(pred_err_t, dim=1)   # [B, T]
    acc = torch.stack(pred_acc_t, dim=1)   # [B, T]  (0/1 floats)
    wp  = word_pos

    out = {}
    out["pred_err_all"] = float(err.mean())
    out["acc_all"]      = float(acc.mean())

    word_mask = (wp >= 0) & (wp <= 2)      # A, B, C → predict B/C/D (within-word)
    bg_mask   = (wp == -1)
    if word_mask.any():
        out["pred_err_word"] = float(err[word_mask].mean())
        out["acc_word"]      = float(acc[word_mask].mean())
    else:
        out["pred_err_word"] = float("nan")
        out["acc_word"]      = float("nan")
    if bg_mask.any():
        out["pred_err_bg"] = float(err[bg_mask].mean())
        out["acc_bg"]      = float(acc[bg_mask].mean())
    else:
        out["pred_err_bg"] = float("nan")
        out["acc_bg"]      = float("nan")

    # Per-transition accuracy
    for label, pos in [("acc_AB", 0), ("acc_BC", 1), ("acc_CD", 2), ("acc_Dbg", 3)]:
        m = (wp == pos)
        out[label] = float(acc[m].mean()) if m.any() else float("nan")

    return out


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(
    num_epochs:    int   = 30,   # was 50; 30 is enough, avoids overtraining
    batch_size:    int   = 64,
    T:             int   = 400,
    p_word_onset:  float = 0.05,
    min_gap:       int   = 6,
    device:        str   = None,
    ckpt_path:     str   = "hipcat_trained.pt",
    seed:          int   = 0,
    # lesion config (passed straight to model.set_lesion / set_pathway)
    lesion_msp:    bool  = False,
    lesion_tsp:    bool  = False,
    lesion_pathways: list[str] | None = None,
    # model hyperparams
    n_dg:          int   = 400,
    n_ca3:         int   = 80,
    n_ca1:         int   = 100,
    settle_steps:  int | None = None,   # None -> hip-cat 30/50/20 cycles
    initial_cycles:int   = 1,
    state_carry:   float = 0.0,          # hip-cat decay.event=1
    predictive_ca3:bool  = False,        # auto-associative CA3->CA3 as in hip-cat
    bigloop_from_prev:bool = True,
    theta_gates_ca3:bool = True,         # mossy suppressed in RECALL (v5)
    # TSP learning-rate overrides (None -> model defaults 0.05/0.05/0.05/0.05)
    lr_ecin_dg:    float | None = None,
    lr_ecin_ca3:   float | None = None,
    lr_ca3_ca3:    float | None = None,
    lr_ca3_ca1:    float | None = None,
    eval_readout:  str   = "auto",       # "auto" = MSP-phase prediction (default); "recall" = CA3-driven
    verbose:       bool  = True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[train] device={device}  vocab={sound_size}  epochs={num_epochs}")
        if lesion_msp or lesion_tsp or lesion_pathways:
            print(f"        LESION: msp={lesion_msp} tsp={lesion_tsp} "
                  f"pathways={lesion_pathways}")

    model_kwargs = dict(
        input_size    = sound_size,
        n_dg          = n_dg, n_ca3 = n_ca3, n_ca1 = n_ca1,
        settle_steps  = settle_steps,
        initial_cycles= initial_cycles,
        state_carry   = state_carry,
        predictive_ca3= predictive_ca3,
        bigloop_from_prev= bigloop_from_prev,
        theta_gates_ca3= theta_gates_ca3,
        device        = device,
        seed          = seed,
        eval_readout  = eval_readout,
    )
    # Only pass TSP lr overrides if explicitly supplied, else use model defaults
    for k, v in [("lr_ecin_dg", lr_ecin_dg),
                 ("lr_ecin_ca3", lr_ecin_ca3),
                 ("lr_ca3_ca3",  lr_ca3_ca3),
                 ("lr_ca3_ca1",  lr_ca3_ca1)]:
        if v is not None:
            model_kwargs[k] = v

    model = HipCatModel(**model_kwargs).to(device)

    if lesion_msp or lesion_tsp:
        model.set_lesion(msp=lesion_msp, tsp=lesion_tsp)
    if lesion_pathways:
        for p in lesion_pathways:
            model.set_pathway(p, on=False)

    history = []   # list of dicts, one per epoch

    chance = 1.0 / ALPHABET_SIZE
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        seq, prev_seq, next_seq, word_pos, _ = make_embedded_stream(
            batch_size=batch_size, T=T,
            p_word_onset=p_word_onset, min_gap=min_gap,
            seed=seed * 10_000 + epoch,
        )
        seq      = seq.to(device)
        prev_seq = prev_seq.to(device)
        next_seq = next_seq.to(device)

        model.reset_state()   # one stream = one event
        pred_err_t = []
        pred_acc_t = []

        for t in range(T):
            cur  = seq[:, t]
            prev = prev_seq[:, t]
            nxt  = next_seq[:, t]

            err, pidx, _ = model.train_step(cur, prev, nxt)
            pred_err_t.append(err.cpu())
            acc = (pidx == nxt).float().cpu()
            pred_acc_t.append(acc)

        m = epoch_metrics(pred_err_t, pred_acc_t, word_pos)
        m["epoch"] = epoch
        history.append(m)

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == num_epochs):
            elapsed = time.time() - t0
            print(f"  ep {epoch:3d}/{num_epochs}  "
                  f"err_word={m['pred_err_word']:.4f}  err_bg={m['pred_err_bg']:.4f}  "
                  f"acc[A→B={m['acc_AB']:.2f} B→C={m['acc_BC']:.2f} "
                  f"C→D={m['acc_CD']:.2f} bg={m['acc_bg']:.2f}]  "
                  f"({elapsed:.1f}s)")

    # ── Save checkpoint ────────────────────────────────────────────────────
    config = dict(
        input_size=sound_size, n_dg=n_dg, n_ca3=n_ca3, n_ca1=n_ca1,
        settle_steps=settle_steps, initial_cycles=initial_cycles,
        state_carry=state_carry, predictive_ca3=predictive_ca3,
        bigloop_from_prev=bigloop_from_prev, theta_gates_ca3=theta_gates_ca3,
        seed=seed,
        lesion_msp=lesion_msp, lesion_tsp=lesion_tsp,
        lesion_pathways=lesion_pathways or [],
        p_word_onset=p_word_onset, min_gap=min_gap,
        T=T, batch_size=batch_size, num_epochs=num_epochs,
        # Also save TSP lrs used, so analysis can reconstruct the model faithfully
        lr_ecin_dg  = model.ecin_to_dg.lr,
        lr_ecin_ca3 = model.ecin_to_ca3.lr,
        lr_ca3_ca3  = model.ca3_to_ca3.lr,
        lr_ca3_ca1  = model.ca3_to_ca1.lr,
        eval_readout = model.eval_readout,
        slot_code = model.slot_code, rec_recall_mult = model.rec_recall_mult,
        rec_recall_mult_eval = model.rec_recall_mult_eval,
        mossy_recall_mult = model.mossy_recall_mult,
        hebb_ca3_ca3 = model.ca3_to_ca3.hebb_mix,
    )
    torch.save({
        "state_dict": model.state_dict(),
        "config":     config,
        "history":    history,
        # for backward compat with older analysis
        "hist_all":   [h["pred_err_all"]  for h in history],
        "hist_word":  [h["pred_err_word"] for h in history],
        "hist_bg":    [h["pred_err_bg"]   for h in history],
    }, ckpt_path)
    if verbose:
        print(f"\n  ✓ saved → {ckpt_path}")

    # ── Training-curve figure ──────────────────────────────────────────────
    fig_path = os.path.splitext(ckpt_path)[0] + "_training_curve.png"
    _plot_training_curve(history, fig_path, chance)
    if verbose:
        print(f"  ✓ saved → {fig_path}\n")

    return model, history


def _plot_training_curve(history, path, chance):
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(ep, [h["pred_err_all"]  for h in history], lw=2, color="#555",
            label="all items")
    ax.plot(ep, [h["pred_err_word"] for h in history], lw=2, ls="--",
            color="#1a6fc4", label="A→B / B→C / C→D")
    ax.plot(ep, [h["pred_err_bg"]   for h in history], lw=2, ls=":",
            color="#c44b1a", label="bg → bg")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Forward-prediction MSE")
    ax.set_title("Prediction error by transition type")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for label, color in [("acc_AB", "#e07b39"),
                         ("acc_BC", "#3a9bd5"),
                         ("acc_CD", "#3ac47d"),
                         ("acc_Dbg","#9b5ea0"),
                         ("acc_bg", "#888")]:
        vals = [h[label] for h in history]
        ax.plot(ep, vals, lw=2, color=color, label=label)
    ax.axhline(chance, lw=1.5, ls="--", color="red", label=f"chance ({chance:.2f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Argmax accuracy")
    ax.set_title("Per-transition forward-prediction accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--T",      type=int, default=400)
    ap.add_argument("--ckpt",   type=str, default="hipcat_trained.pt")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--state_carry", type=float, default=0.0,
                    help="1 - decay.event: fraction of layer activity carried "
                         "into the next pip. hip-cat default 0.0 (full reset).")
    ap.add_argument("--hipcat_ca3", action="store_true",
                    help="Strict hip-cat CA3: mossy rel=8 in every phase, no recall-phase "
                         "gating (auto-associative CA3->CA3 is already the default).")
    ap.add_argument("--hetero_ca3", action="store_true",
                    help="Heteroassociative CA3->CA3 (previous tone's CA3 drives the current "
                         "one). Off by default; the default model completes an omitted tone "
                         "via the big loop + auto-associative pattern completion.")
    ap.add_argument("--lesion_msp",       action="store_true")
    ap.add_argument("--lesion_tsp",       action="store_true")
    ap.add_argument("--lesion_pathways",  type=str, nargs="*", default=None,
                    help="Per-pathway lesions, e.g. --lesion_pathways ca3_ca3 dg_ca3")
    # TSP learning-rate overrides (default: .proj values 0.2/0.2/0.2/0.05)
    ap.add_argument("--lr_ecin_dg",  type=float, default=None,
                    help="Override TSP EC_in->DG learning rate (default 0.05)")
    ap.add_argument("--lr_ecin_ca3", type=float, default=None,
                    help="Override TSP EC_in->CA3 learning rate (default 0.05)")
    ap.add_argument("--lr_ca3_ca3",  type=float, default=None,
                    help="Override TSP CA3->CA3 recurrent lr (default 0.05)")
    ap.add_argument("--lr_ca3_ca1",  type=float, default=None,
                    help="Override TSP CA3->CA1 lr (.proj 0.05)")
    ap.add_argument("--eval_readout", type=str, default="auto", choices=["auto", "recall"],
                    help="Phase the prediction (and the big-loop feedback) is read from. "
                         "'auto' (default) = MSP phase, as in the dissertation. 'recall' = "
                         "CA3-driven; unstable for B->C and at chance under a TSP lesion.")
    ap.add_argument("--lr_tsp_scale", type=float, default=None,
                    help="Shortcut: multiply ALL TSP lrs by this factor. "
                         "E.g. --lr_tsp_scale 0.5 halves them all.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device

    # Resolve TSP lr overrides. --lr_tsp_scale is applied on top of model defaults
    # if no per-projection override is given.
    lr_ecin_dg  = args.lr_ecin_dg
    lr_ecin_ca3 = args.lr_ecin_ca3
    lr_ca3_ca3  = args.lr_ca3_ca3
    lr_ca3_ca1  = args.lr_ca3_ca1
    if args.lr_tsp_scale is not None:
        s = args.lr_tsp_scale
        if lr_ecin_dg  is None: lr_ecin_dg  = 0.05 * s
        if lr_ecin_ca3 is None: lr_ecin_ca3 = 0.05 * s
        if lr_ca3_ca3  is None: lr_ca3_ca3  = 0.05 * s
        if lr_ca3_ca1  is None: lr_ca3_ca1  = HIPCAT["lr_ca3_ca1"]  * s

    train_model(
        num_epochs      = args.epochs,
        batch_size      = args.batch,
        T               = args.T,
        ckpt_path       = args.ckpt,
        device          = device,
        seed            = args.seed,
        lesion_msp      = args.lesion_msp,
        lesion_tsp      = args.lesion_tsp,
        lesion_pathways = args.lesion_pathways,
        state_carry     = args.state_carry,
        predictive_ca3    = args.hetero_ca3,
        bigloop_from_prev = True,
        theta_gates_ca3   = not args.hipcat_ca3,
        lr_ecin_dg      = lr_ecin_dg,
        lr_ecin_ca3     = lr_ecin_ca3,
        lr_ca3_ca3      = lr_ca3_ca3,
        lr_ca3_ca1      = lr_ca3_ca1,
        eval_readout    = args.eval_readout,
    )
