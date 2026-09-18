

import os
import argparse
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model    import HipCatModel
from data     import (make_embedded_stream, make_random_stream,
                      sound_size, WORD, WORD_LEN, ALPHABET_SIZE)
from training import train_model

WORD_LABELS    = ["A", "B", "C", "D"]
TRANS_LABELS   = ["A→B", "B→C", "C→D", "D→bg", "bg→bg"]
TRANS_KEYS     = ["acc_AB", "acc_BC", "acc_CD", "acc_Dbg", "acc_bg"]
STYLE          = dict(fontsize=11, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def ensure(d): os.makedirs(d, exist_ok=True)

def save(fig, path):
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")

def to_np(x):
    if isinstance(x, np.ndarray): return x
    return x.detach().cpu().numpy()

def sem(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0: return 0.0
    return float(x.std() / np.sqrt(len(x)))


def overlap_sim(a: np.ndarray, b: np.ndarray, thr: float = 0.01) -> float:
    """
    Sparse-code overlap similarity (cosine of binarised vectors).
    Equivalent to |A∩B| / sqrt(|A|·|B|).
    Returns 0 if either vector has no active units.
    """
    A = (a > thr).astype(float)
    B = (b > thr).astype(float)
    nA, nB = A.sum(), B.sum()
    if nA == 0 or nB == 0:
        return 0.0
    return float((A * B).sum() / np.sqrt(nA * nB))


def overlap_rsm(vecs: np.ndarray, thr: float = 0.01) -> np.ndarray:
    """vecs: [N, units] → [N, N] overlap similarity matrix."""
    N = vecs.shape[0]
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = overlap_sim(vecs[i], vecs[j], thr=thr)
    return M


def corr_safe(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.corrcoef(X), nan=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint / model helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in ck:
        ck = {"state_dict": ck, "config": {}, "history": []}
    return ck

def build_model(cfg: dict, device: str) -> HipCatModel:
    kwargs = dict(
        input_size    = cfg.get("input_size",    sound_size),
        n_dg          = cfg.get("n_dg",          400),
        n_ca3         = cfg.get("n_ca3",         80),
        n_ca1         = cfg.get("n_ca1",         100),
        settle_steps  = cfg.get("settle_steps",  None),   # None -> hip-cat 30/50/20 cycles
        eval_readout  = cfg.get("eval_readout", "recall"),
        initial_cycles= cfg.get("initial_cycles",1),
        device        = device,
        seed          = cfg.get("seed",          0),
    )
    # Pass TSP lrs only if present in config (older checkpoints won't have them).
    for k in ["lr_ecin_dg", "lr_ecin_ca3", "lr_ca3_ca3", "lr_ca3_ca1"]:
        if k in cfg:
            kwargs[k] = cfg[k]
    return HipCatModel(**kwargs).to(device)

def restore(model, state_dict):
    miss, unex = model.load_state_dict(state_dict, strict=False)
    if miss:  print(f"    [warn] missing keys: {len(miss)}")
    if unex:  print(f"    [warn] unexpected keys: {len(unex)}")


# ══════════════════════════════════════════════════════════════════════════════
# Run a stream and collect everything we need
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_stream(model: HipCatModel,
               seq, prev_seq, next_seq,
               capture_layers: bool = True):
    """
    Runs eval_step on every timestep. Returns dict with:
        pred_err   [B, T]
        pred_idx   [B, T]
        acc        [B, T]
        seq        [B, T] (numpy)
        next_seq   [B, T] (numpy)
        prev_seq   [B, T] (numpy)
      if capture_layers:
        dg_init  ca3_init  ca1_init  ecout_init  ecin_init   each [B, T, units]
        dg_set   ca3_set   ca1_set   ecout_set   ecin_set    each [B, T, units]
    """
    device = model.device
    B, T   = seq.shape

    pred_err = torch.zeros(B, T)
    pred_idx = torch.zeros(B, T, dtype=torch.long)

    rec = {k: [] for k in
           ["dg_init","ca3_init","ca1_init","ecout_init","ecin_init",
            "dg_set", "ca3_set", "ca1_set", "ecout_set", "ecin_set"]}

    seq_d  = seq.to(device)
    prev_d = prev_seq.to(device)
    next_d = next_seq.to(device)

    for t in range(T):
        err, pidx, snap = model.eval_step(
            seq_d[:, t], prev_d[:, t], next_d[:, t],
            capture_initial=capture_layers,
        )
        pred_err[:, t] = err.cpu()
        pred_idx[:, t] = pidx.cpu()

        if capture_layers:
            ini  = snap.get("initial", snap["settled"])
            sett = snap["settled"]
            rec["dg_init"].append(ini["DG"].cpu().numpy())
            rec["ca3_init"].append(ini["CA3"].cpu().numpy())
            rec["ca1_init"].append(ini["CA1"].cpu().numpy())
            rec["ecout_init"].append(ini["ECout"].cpu().numpy())
            rec["ecin_init"].append(ini["ECin"].cpu().numpy())
            rec["dg_set"].append(sett["DG"].cpu().numpy())
            rec["ca3_set"].append(sett["CA3"].cpu().numpy())
            rec["ca1_set"].append(sett["CA1"].cpu().numpy())
            rec["ecout_set"].append(sett["ECout"].cpu().numpy())
            rec["ecin_set"].append(sett["ECin"].cpu().numpy())

    out = dict(
        pred_err = pred_err.numpy(),
        pred_idx = pred_idx.numpy(),
        acc      = (pred_idx == next_seq).float().numpy(),
        seq      = seq.numpy(),
        prev_seq = prev_seq.numpy(),
        next_seq = next_seq.numpy(),
    )
    if capture_layers:
        # Stack list of [B, units] over T → [T, B, units] → transpose to [B, T, units]
        for k, v in rec.items():
            arr = np.stack(v, axis=0)            # [T, B, units]
            out[k] = np.transpose(arr, (1, 0, 2)) # [B, T, units]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 – Training curves
# ══════════════════════════════════════════════════════════════════════════════

def fig_training_curves(ck, figdir):
    history = ck.get("history", [])
    if not history:
        print("  [skip fig1] no per-epoch history; checkpoint may be old format")
        return
    ep = [h["epoch"] for h in history]
    chance = 1.0 / ALPHABET_SIZE

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training curves (faithful HipCat)", **STYLE)

    ax = axes[0]
    ax.plot(ep, [h["pred_err_all"]  for h in history], lw=2, color="#555",
            label="all items")
    ax.plot(ep, [h["pred_err_word"] for h in history], lw=2, ls="--",
            color="#1a6fc4", label="A→B / B→C / C→D")
    ax.plot(ep, [h["pred_err_bg"]   for h in history], lw=2, ls=":",
            color="#c44b1a", label="bg → bg")
    ax.set_xlabel("Epoch", **STYLE)
    ax.set_ylabel("Forward-prediction MSE", **STYLE)
    ax.set_title("Prediction error by transition type")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    palette = {"acc_AB":"#e07b39","acc_BC":"#3a9bd5","acc_CD":"#3ac47d",
               "acc_Dbg":"#9b5ea0","acc_bg":"#888"}
    for key, lbl in zip(TRANS_KEYS, TRANS_LABELS):
        ax.plot(ep, [h[key] for h in history], lw=2, color=palette[key], label=lbl)
    ax.axhline(chance, lw=1.5, ls="--", color="red", label=f"chance ({chance:.2f})")
    ax.set_xlabel("Epoch", **STYLE)
    ax.set_ylabel("Argmax accuracy", **STYLE)
    ax.set_title("Per-transition forward-prediction accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig01_training_curves.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 – Per-position accuracy at end of training (eval pass)
# ══════════════════════════════════════════════════════════════════════════════

def fig_position_accuracy(results, word_pos, figdir, tag=""):
    wp  = word_pos
    acc = results["acc"]
    means, sems_, counts = [], [], []
    for gid in [-1, 0, 1, 2, 3]:
        m = (wp == gid)
        v = acc[m]
        means.append(float(v.mean())  if len(v) else np.nan)
        sems_.append(sem(v)           if len(v) else 0.0)
        counts.append(len(v))

    chance = 1.0 / ALPHABET_SIZE
    labels = ["bg→bg", "A→B", "B→C", "C→D", "D→bg"]
    colors = ["#888","#e07b39","#3a9bd5","#3ac47d","#9b5ea0"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=sems_, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.8)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2,
                b.get_height() + 0.02, f"n={c}",
                ha="center", fontsize=8, color="#444")
    ax.axhline(chance, lw=1.5, ls="--", color="red", label=f"chance ({chance:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("P(ECout argmax = next item)", **STYLE)
    title = "Forward-prediction accuracy by transition type"
    if tag: title += f"  [{tag}]"
    ax.set_title(title, **STYLE)
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fname = "fig02_position_accuracy.png" if not tag \
            else f"fig02_position_accuracy_{tag}.png"
    save(fig, os.path.join(figdir, fname))


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 – Onset-aligned forward-pred accuracy
# ══════════════════════════════════════════════════════════════════════════════

def fig_onset_aligned(results, word_pos, figdir, window=12):
    acc = results["acc"]
    B, T = word_pos.shape
    chance = 1.0 / ALPHABET_SIZE

    traces = []
    for b in range(B):
        zeros = np.where(word_pos[b] == 0)[0]
        for t0 in zeros:
            lo, hi = t0 - window, t0 + window + 1
            if lo >= 0 and hi <= T:
                traces.append(acc[b, lo:hi])
    if not traces:
        print("  [skip fig3] no complete onset windows"); return
    traces = np.array(traces)
    mn = traces.mean(0); se = traces.std(0) / np.sqrt(len(traces))
    x  = np.arange(-window, window + 1)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, lbl in enumerate(WORD_LABELS):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.10, color="#3a9bd5")
        ax.text(i, 1.04, lbl, ha="center", fontsize=11, color="#3a9bd5",
                fontweight="bold")
    ax.axvline(0, lw=1.5, ls="--", color="#3a9bd5", label="A onset")
    ax.axhline(chance, lw=1.5, ls="--", color="red", label=f"chance ({chance:.2f})")
    ax.plot(x, mn, lw=2.5, color="#1a1a2e", label="mean ± SEM")
    ax.fill_between(x, mn - se, mn + se, alpha=0.25, color="#1a1a2e")
    ax.set_xlabel("Time relative to A onset", **STYLE)
    ax.set_ylabel("P(next item correct)", **STYLE)
    ax.set_title("Onset-aligned forward-prediction accuracy\n"
                 "(should rise sharply at B/C/D positions)", **STYLE)
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig03_onset_aligned_accuracy.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 – Linear decoder readout from each layer
# ══════════════════════════════════════════════════════════════════════════════

def fig_decoder_readout(results, word_pos, figdir):
    """
    For each layer, train a logistic-regression decoder to predict the NEXT
    item identity from the layer's settled-response activity. Flattens
    (B, T) into one big set of (timestep, layer_activity) examples and holds
    out 30% for test.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("  [skip fig4] sklearn not available")
        return None, None

    wp_full  = word_pos                # [B, T]
    nxt_full = results["next_seq"]     # [B, T]
    layers = [
        ("DG",    results["dg_set"]),    # [B, T, units]
        ("CA3",   results["ca3_set"]),
        ("CA1",   results["ca1_set"]),
        ("ECout", results["ecout_set"]),
    ]

    # Skip t=0 (no real prev_seq history) and the last col (next_seq wraps).
    B, T = wp_full.shape
    valid_t = slice(1, T - 1)
    wp_v  = wp_full[:, valid_t].reshape(-1)
    y_all = nxt_full[:, valid_t].reshape(-1)

    overall_scores = {}
    per_pos_scores = {}

    for name, X_full in layers:
        X = X_full[:, valid_t, :].reshape(-1, X_full.shape[-1])
        try:
            X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
                X, y_all, np.arange(len(y_all)),
                test_size=0.3, random_state=0, stratify=None,
            )
            clf = LogisticRegression(max_iter=2000, multi_class="auto",
                                     solver="lbfgs", C=1.0)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            overall_scores[name] = float((preds == y_te).mean())

            wp_te = wp_v[idx_te]
            pp = {}
            for gid, lbl in zip([-1, 0, 1, 2, 3],
                                ["bg→bg","A→B","B→C","C→D","D→bg"]):
                m = (wp_te == gid)
                pp[lbl] = float((preds[m] == y_te[m]).mean()) if m.any() else np.nan
            per_pos_scores[name] = pp
        except Exception as e:
            print(f"    [warn] decoder {name} failed: {e}")
            overall_scores[name] = float("nan")
            per_pos_scores[name] = {l: float("nan")
                                    for l in ["bg→bg","A→B","B→C","C→D","D→bg"]}

    chance = 1.0 / ALPHABET_SIZE
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Linear decoder readout from each layer (predict NEXT item)", **STYLE)

    ax = axes[0]
    names = list(overall_scores.keys())
    cols  = ["#3a9bd5","#e07b39","#3ac47d","#9b5ea0"]
    # Mean accuracy on within-word transitions only (A→B, B→C, C→D).
    # The overall-accuracy number is misleading because bg dominates by ~20x
    # and bg is unpredictable by design, so it drags the mean toward chance.
    word_trans_keys = ["A→B", "B→C", "C→D"]
    word_means = [float(np.nanmean([per_pos_scores[n][k] for k in word_trans_keys]))
                  for n in names]
    ax.bar(names, word_means, color=cols, edgecolor="black")
    ax.axhline(chance, lw=1.5, ls="--", color="red", label=f"chance ({chance:.2f})")
    ax.set_ylabel("Mean test accuracy (A→B, B→C, C→D)", **STYLE)
    ax.set_title("Word-transition decoding accuracy per layer\n"
                 "(overall acc would be misleading: bg is ~85% of timesteps and unpredictable)")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.25, axis="y")

    ax = axes[1]
    pos_labels = ["bg→bg","A→B","B→C","C→D","D→bg"]
    x = np.arange(len(pos_labels))
    width = 0.18
    for i, name in enumerate(names):
        vals = [per_pos_scores[name][l] for l in pos_labels]
        ax.bar(x + (i - 1.5) * width, vals, width=width,
               label=name, color=cols[i], edgecolor="black", linewidth=0.5)
    ax.axhline(chance, lw=1.5, ls="--", color="red")
    ax.set_xticks(x); ax.set_xticklabels(pos_labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("Decoder accuracy", **STYLE)
    ax.set_title("Per-transition decoding accuracy")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig04_decoder_readout.png"))

    # Build a richer output dict for the summary
    rich = {}
    for n in names:
        rich[n] = {
            "overall":   overall_scores[n],
            "word_mean": float(np.nanmean(
                [per_pos_scores[n][k] for k in word_trans_keys])),
            "per_pos":   per_pos_scores[n],
        }
    return rich, per_pos_scores


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 – Sparse-code overlap RSM
# ══════════════════════════════════════════════════════════════════════════════

def _build_position_prototypes(acts, wp, n_reps=10):
    """
    For each word position 0..3, build a prototype by averaging the LAST
    n_reps occurrences across the entire batch.
    acts:  [B, T, n_units]   wp: [B, T]
    Returns [4, n_units].
    """
    B, T, U = acts.shape
    out = []
    for pos in range(4):
        bs, ts = np.where(wp == pos)
        if len(bs) == 0:
            out.append(np.zeros(U)); continue
        # take last n_reps occurrences (in flat order)
        n = min(n_reps, len(bs))
        bs, ts = bs[-n:], ts[-n:]
        out.append(acts[bs, ts].mean(0))
    return np.array(out)


def fig_overlap_rsm(results, word_pos, figdir):
    wp = word_pos                       # full [B, T]
    layers = [
        ("DG",    results["dg_set"]),
        ("CA3",   results["ca3_set"]),
        ("CA1",   results["ca1_set"]),
        ("ECout", results["ecout_set"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Sparse-code overlap RSM (settled response, A B C D)\n"
                 "Off-diagonal warm = positions share active units. "
                 "Diagonal should always be 1.", **STYLE)
    for ax, (name, acts) in zip(axes, layers):
        protos = _build_position_prototypes(acts, wp, n_reps=20)
        M = overlap_rsm(protos, thr=0.01)
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="equal")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(WORD_LABELS, fontsize=12, fontweight="bold")
        ax.set_yticklabels(WORD_LABELS, fontsize=12, fontweight="bold")
        ax.set_title(name, **STYLE)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=10, color="white" if M[i,j] < 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="overlap")
    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig05_overlap_rsm.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 – Initial vs Settled response (Schapiro Fig 2 motif)
# ══════════════════════════════════════════════════════════════════════════════

def fig_initial_vs_settled(results, word_pos, figdir):
    wp = word_pos                       # full [B, T]
    layer_pairs = [
        ("DG",    results["dg_init"],    results["dg_set"]),
        ("CA3",   results["ca3_init"],   results["ca3_set"]),
        ("CA1",   results["ca1_init"],   results["ca1_set"]),
        ("ECout", results["ecout_init"], results["ecout_set"]),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("Initial vs settled response  (Schapiro Fig 2 motif)\n"
                 "TOP: cycle 1 (initial response, before recurrence acts). "
                 "BOTTOM: cycle 30 (settled).\n"
                 "Sequential structure should grow as the network settles.", **STYLE)

    for col, (name, A_init, A_set) in enumerate(layer_pairs):
        for row, (acts, label) in enumerate(
            [(A_init, "initial"), (A_set, "settled")]):
            protos = _build_position_prototypes(acts, wp, n_reps=20)
            M = overlap_rsm(protos, thr=0.01)
            ax = axes[row, col]
            im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="equal")
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(WORD_LABELS, fontsize=11, fontweight="bold")
            ax.set_yticklabels(WORD_LABELS, fontsize=11, fontweight="bold")
            ax.set_title(f"{name} ({label})", fontsize=11, fontweight="bold")
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            fontsize=9,
                            color="white" if M[i,j] < 0.5 else "black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig06_initial_vs_settled.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 – Confusion matrix
# ══════════════════════════════════════════════════════════════════════════════

def fig_confusion(results, word_pos, figdir):
    wp = word_pos
    pred = results["pred_idx"]
    word_set = set(WORD)
    conf = np.zeros((4, 5))
    for b in range(wp.shape[0]):
        for t in range(wp.shape[1]):
            pos = int(wp[b, t])
            if pos < 0: continue
            p = int(pred[b, t])
            if p in word_set:
                conf[pos, WORD.index(p)] += 1
            else:
                conf[pos, 4] += 1
    row_sums = conf.sum(axis=1, keepdims=True).clip(min=1)
    conf_norm = conf / row_sums

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(conf_norm, vmin=0, vmax=1, cmap="Reds", aspect="equal")
    col_labels = WORD_LABELS + ["bg/other"]
    ax.set_xticks(range(5)); ax.set_yticks(range(4))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels(WORD_LABELS, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted item", **STYLE)
    ax.set_ylabel("Current word position", **STYLE)
    ax.set_title("Confusion matrix: argmax(ECout) at each word position\n"
                 "(A row should peak at B; B at C; C at D; D anywhere)", **STYLE)
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f"{conf_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=10,
                    color="white" if conf_norm[i,j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(predict | row)")
    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig07_confusion.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 – Rasters
# ══════════════════════════════════════════════════════════════════════════════

def fig_rasters(results, word_pos, figdir, t_show=80):
    wp     = word_pos[0]                              # one batch member
    t_show = min(t_show, len(wp))
    layers = [("DG",    results["dg_set"][0],    "#3a9bd5"),
              ("CA3",   results["ca3_set"][0],   "#e07b39"),
              ("CA1",   results["ca1_set"][0],   "#3ac47d"),
              ("ECout", results["ecout_set"][0], "#9b5ea0")]

    fig, axes = plt.subplots(len(layers), 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Population activity rasters (settled response, first {t_show} steps)\n"
                 "Shaded = within-word (ABCD)", **STYLE)
    for ax, (name, acts, col) in zip(axes, layers):
        a = acts[:t_show]
        n = a.shape[1]
        if n > 200:
            a = a[:, ::n//200]
        ax.imshow(a.T, aspect="auto", cmap="hot", interpolation="nearest",
                  origin="lower", extent=[0, t_show, 0, a.shape[1]])
        in_w = False; t_s = 0
        for t in range(t_show):
            if wp[t] >= 0 and not in_w:
                in_w = True; t_s = t
            elif wp[t] < 0 and in_w:
                in_w = False
                ax.axvspan(t_s, t, alpha=0.18, color=col)
        if in_w:
            ax.axvspan(t_s, t_show, alpha=0.18, color=col)
        ax.set_ylabel(name, **STYLE); ax.set_yticks([])

    ax = axes[-1]
    _, top = ax.get_ylim()
    for t in range(t_show):
        p = int(wp[t])
        if p >= 0:
            ax.text(t + 0.5, -top * 0.08, WORD_LABELS[p],
                    ha="center", fontsize=7, color="#222", clip_on=False)
    ax.set_xlabel("Time step", **STYLE)
    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig08_rasters.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9 – Weight distributions
# ══════════════════════════════════════════════════════════════════════════════

def fig_weight_distributions(model: HipCatModel, figdir):
    conns = [
        ("EC_in→CA1 (MSP)",      lambda m: m.ecin_to_ca1.weight),
        ("CA1→EC_out (MSP)",     lambda m: m.ca1_to_ecout.weight),
        ("EC_out→CA1 (big-loop)",lambda m: m.ecout_to_ca1.weight),
        ("EC_in→DG (TSP)",       lambda m: m.ecin_to_dg.weight * m.ecin_to_dg.mask),
        ("EC_in→CA3 (TSP)",      lambda m: m.ecin_to_ca3.weight * m.ecin_to_ca3.mask),
        ("DG→CA3 (Mossy fixed)", lambda m: m.dg_to_ca3.weight),
        ("CA3→CA3 (recurrent)",  lambda m: m.ca3_to_ca3.weight),
        ("CA3→CA1",              lambda m: m.ca3_to_ca1.weight),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    fig.suptitle("Weight distributions per projection\n"
                 "Mossy stays near init (lr=0). Spike at 1.0 = saturation.", **STYLE)
    for ax, (name, getter) in zip(axes.ravel(), conns):
        W = getter(model).detach().cpu().numpy().ravel()
        W = W[W > 1e-7]
        if len(W) == 0:
            ax.set_visible(False); continue
        ax.hist(W, bins=40, color="#3a9bd5", edgecolor="white", linewidth=0.4)
        ax.axvline(W.mean(), color="red", lw=1.5, ls="--",
                   label=f"μ={W.mean():.3f}")
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("weight", fontsize=8)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig09_weight_distributions.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Fig 10 / 11 – Lesion experiments
# ══════════════════════════════════════════════════════════════════════════════

LESION_CONDITIONS = [
    # (label, msp, tsp, extra_pathways)
    ("Intact",          False, False, []),
    ("MSP lesion",      True,  False, []),
    ("TSP lesion",      False, True,  []),
    ("DG lesion",       False, False, ["ecin_dg", "dg_ca3"]),
    ("CA3 recur. off",  False, False, ["ca3_ca3"]),
    ("CA3→CA1 off",     False, False, ["ca3_ca1"]),
    ("Big-loop off",    False, False, ["bigloop"]),
]


def run_lesion_train_sweep(figdir, base_args):
    """
    Train one fresh model per lesion condition. Save each as a checkpoint.
    Returns dict label -> (history list, eval results, model state_dict path).
    """
    results = {}
    for (label, lm, lt, paths) in LESION_CONDITIONS:
        ckpt = os.path.join(figdir, "lesions",
                            f"hipcat_{label.replace(' ','_').replace('→','-')}.pt")
        ensure(os.path.dirname(ckpt))
        print(f"\n[lesion train] {label}  (msp={lm} tsp={lt} paths={paths})")
        t0 = time.time()
        _, history = train_model(
            num_epochs      = base_args["epochs"],
            batch_size      = base_args["batch"],
            T               = base_args["T"],
            ckpt_path       = ckpt,
            device          = base_args["device"],
            seed            = base_args["seed"],
            lesion_msp      = lm,
            lesion_tsp      = lt,
            lesion_pathways = paths if paths else None,
            verbose         = False,
        )
        dt = time.time() - t0
        # Quick final eval on a fresh test stream
        ck = torch.load(ckpt, map_location=base_args["device"], weights_only=False)
        model = build_model(ck.get("config", {}), base_args["device"])
        restore(model, ck["state_dict"])
        if lm or lt: model.set_lesion(msp=lm, tsp=lt)
        for p in paths: model.set_pathway(p, on=False)
        model.eval()
        seq, prev, nxt, wp, _ = make_embedded_stream(
            batch_size=32, T=400, seed=base_args["seed"] + 9999)
        eval_res = run_stream(model, seq, prev, nxt, capture_layers=False)
        results[label] = dict(history=history, eval=eval_res, wp=wp.numpy(), ckpt=ckpt)
        print(f"  done in {dt:.1f}s.  final acc B→C={history[-1]['acc_BC']:.2f}  "
              f"C→D={history[-1]['acc_CD']:.2f}")
    return results


def fig_lesion_train(lesion_results, figdir):
    chance = 1.0 / ALPHABET_SIZE
    labels = list(lesion_results.keys())

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("Training-time lesion sweep\n"
                 "Top: per-transition accuracy (final epoch). "
                 "Bottom: A→B / B→C / C→D learning curves.",
                 **STYLE)

    # Panel 1: bars per transition × condition
    ax = axes[0]
    bar_w = 0.13
    x = np.arange(5)   # bg, A→B, B→C, C→D, D→bg
    pos_keys = ["acc_bg", "acc_AB", "acc_BC", "acc_CD", "acc_Dbg"]
    pos_lbls = ["bg→bg", "A→B", "B→C", "C→D", "D→bg"]

    cond_colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for i, label in enumerate(labels):
        h = lesion_results[label]["history"][-1]
        vals = [h[k] for k in pos_keys]
        ax.bar(x + (i - len(labels)/2 + 0.5) * bar_w, vals, width=bar_w,
               label=label, color=cond_colors[i], edgecolor="black", linewidth=0.4)
    ax.axhline(chance, lw=1.5, ls="--", color="red", label=f"chance ({chance:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(pos_lbls, fontsize=11, fontweight="bold")
    ax.set_ylabel("Forward-prediction accuracy", **STYLE)
    ax.set_title("Final-epoch accuracy by lesion condition")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8, ncol=4); ax.grid(True, alpha=0.25, axis="y")

    # Panel 2: B→C and C→D learning curves
    ax = axes[1]
    ls_cycle = ["-", "--", "-."]
    for i, label in enumerate(labels):
        history = lesion_results[label]["history"]
        ep = [h["epoch"] for h in history]
        for j, key in enumerate(["acc_BC", "acc_CD"]):
            vals = [h[key] for h in history]
            lab  = f"{label} {key[-2:]}"
            ax.plot(ep, vals, lw=1.7, ls=ls_cycle[j],
                    color=cond_colors[i], label=lab if j == 0 else None,
                    alpha=0.85)
    ax.axhline(chance, lw=1.2, ls=":", color="red")
    ax.set_xlabel("Epoch", **STYLE)
    ax.set_ylabel("Accuracy", **STYLE)
    ax.set_title("B→C (solid) and C→D (dashed) learning curves")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8, ncol=4); ax.grid(True, alpha=0.25)

    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig10_lesion_train.png"))


def fig_lesion_test(intact_ckpt_path, figdir, base_args):
    """
    Train-time intact, test-time-only ablation. Compares Intact, MSP-off, TSP-off
    on an identical evaluation stream.
    """
    chance = 1.0 / ALPHABET_SIZE
    seq, prev, nxt, wp, _ = make_embedded_stream(
        batch_size=32, T=400, seed=base_args["seed"] + 12345)
    wp_np = wp.numpy()

    conditions = [
        ("Intact",     False, False, []),
        ("MSP off",    True,  False, []),
        ("TSP off",    False, True,  []),
        ("Big-loop off",False,False, ["bigloop"]),
    ]
    rows = []
    for (label, lm, lt, paths) in conditions:
        ck = torch.load(intact_ckpt_path,
                        map_location=base_args["device"], weights_only=False)
        model = build_model(ck.get("config", {}), base_args["device"])
        restore(model, ck["state_dict"])
        if lm or lt: model.set_lesion(msp=lm, tsp=lt)
        for p in paths: model.set_pathway(p, on=False)
        model.eval()
        res = run_stream(model, seq, prev, nxt, capture_layers=False)
        means = []
        for gid in [-1, 0, 1, 2, 3]:
            m = (wp_np == gid)
            v = res["acc"][m]
            means.append(float(v.mean()) if len(v) else np.nan)
        rows.append((label, means))

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Test-time-only ablation (model trained intact, ablated at test)",
                 **STYLE)
    x = np.arange(5)
    bar_w = 0.20
    cond_colors = ["#1a6fc4","#c44b1a","#2a8a4a","#9b5ea0"]
    pos_lbls = ["bg→bg","A→B","B→C","C→D","D→bg"]
    for i, (label, vals) in enumerate(rows):
        ax.bar(x + (i - 1.5) * bar_w, vals, width=bar_w, label=label,
               color=cond_colors[i], edgecolor="black", linewidth=0.5)
    ax.axhline(chance, lw=1.5, ls="--", color="red", label=f"chance ({chance:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(pos_lbls, fontsize=11, fontweight="bold")
    ax.set_ylabel("Forward-prediction accuracy", **STYLE)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig11_lesion_test.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Fig 12 – Confusion matrices per lesion condition
# ══════════════════════════════════════════════════════════════════════════════

def _compute_confusion(pred_idx, word_pos):
    """Compute confusion matrix: rows = word position (A,B,C,D),
    cols = predicted item (A,B,C,D,bg/other). Returns [4, 5] normalised."""
    word_set = set(WORD)
    conf = np.zeros((4, 5))
    B, T = word_pos.shape
    for b in range(B):
        for t in range(T):
            pos = int(word_pos[b, t])
            if pos < 0:
                continue
            p = int(pred_idx[b, t])
            if p in word_set:
                conf[pos, WORD.index(p)] += 1
            else:
                conf[pos, 4] += 1
    row_sums = conf.sum(axis=1, keepdims=True).clip(min=1)
    return conf / row_sums


def fig_lesion_confusion(intact_ckpt_path, figdir, base_args):
    """
    Confusion matrix for each of 4 lesion conditions.
    Shows HOW predictions change under each lesion, not just accuracy.
    """
    conditions = [
        ("Intact",       False, False, []),
        ("MSP off",      True,  False, []),
        ("TSP off",      False, True,  []),
        ("Big-loop off", False, False, ["bigloop"]),
    ]

    seq, prev, nxt, wp, _ = make_embedded_stream(
        batch_size=32, T=400, seed=base_args["seed"] + 7777)
    wp_np = wp.numpy()

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Confusion matrices per lesion condition\n"
                 "Rows = current word position; Cols = predicted item. "
                 "Ideal: A→B, B→C, C→D on the super-diagonal.", **STYLE)

    col_labels = WORD_LABELS + ["bg/other"]

    for ax, (label, lm, lt, paths) in zip(axes, conditions):
        ck = torch.load(intact_ckpt_path,
                        map_location=base_args["device"], weights_only=False)
        model = build_model(ck.get("config", {}), base_args["device"])
        restore(model, ck["state_dict"])
        if lm or lt:
            model.set_lesion(msp=lm, tsp=lt)
        for p in paths:
            model.set_pathway(p, on=False)
        model.eval()

        res = run_stream(model, seq, prev, nxt, capture_layers=False)
        conf = _compute_confusion(res["pred_idx"], wp_np)

        im = ax.imshow(conf, vmin=0, vmax=1, cmap="Reds", aspect="equal")
        ax.set_xticks(range(5)); ax.set_yticks(range(4))
        ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
        ax.set_yticklabels(WORD_LABELS, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted item")
        ax.set_ylabel("Current position" if label == "Intact" else "")
        ax.set_title(label, **STYLE)
        for i in range(4):
            for j in range(5):
                ax.text(j, i, f"{conf[i,j]:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if conf[i,j] > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig12_lesion_confusion.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Fig 13 – Gaussian noise robustness
# ══════════════════════════════════════════════════════════════════════════════

def fig_noise_robustness(intact_ckpt_path, figdir, base_args,
                          noise_levels=None):
    """
    Add Gaussian noise to the EC_in input representation at test time.
    Sweep noise σ from 0 to 1.0 and measure within-motif prediction accuracy
    (A→B, B→C, C→D averaged) for each lesion condition.

    This tests how robust the model's predictions are to degraded/noisy input,
    simulating imprecise sensory encoding.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    conditions = [
        ("Intact",       False, False, []),
        ("MSP off",      True,  False, []),
        ("TSP off",      False, True,  []),
        ("Big-loop off", False, False, ["bigloop"]),
    ]
    cond_colors = ["#1a6fc4", "#c44b1a", "#2a8a4a", "#9b5ea0"]

    seq, prev, nxt, wp, _ = make_embedded_stream(
        batch_size=32, T=400, seed=base_args["seed"] + 5555)
    wp_np = wp.numpy()
    device = base_args["device"]

    # We need to intercept the input to add noise. We'll do this by overriding
    # make_ecin_input temporarily. Instead, we create noisy versions of the
    # prev_seq and seq and run them through the standard pipeline, adding noise
    # to the one-hot representation BEFORE it enters the model.

    # Build the clean one-hot inputs for all timesteps
    B, T = seq.shape

    # within-motif mask: positions 0,1,2 (A,B,C → predict B,C,D)
    motif_within = ((wp_np >= 0) & (wp_np <= 2))

    results = {c[0]: [] for c in conditions}

    for sigma in noise_levels:
        for (label, lm, lt, paths) in conditions:
            ck = torch.load(intact_ckpt_path, map_location=device,
                            weights_only=False)
            model = build_model(ck.get("config", {}), device)
            restore(model, ck["state_dict"])
            if lm or lt:
                model.set_lesion(msp=lm, tsp=lt)
            for p in paths:
                model.set_pathway(p, on=False)
            model.eval()

            # Run timestep by timestep with noise injection
            rng = np.random.RandomState(base_args["seed"] + int(sigma * 1000))
            pred_correct = np.zeros((B, T), dtype=bool)

            for t in range(T):
                cur = seq[:, t].to(device)
                prv = prev[:, t].to(device)
                target = nxt[:, t].to(device)

                # Build clean input
                ecin_clean = model.make_ecin_input(cur, prv)

                # Add Gaussian noise
                if sigma > 0:
                    noise = torch.randn_like(ecin_clean) * sigma
                    ecin_noisy = (ecin_clean + noise).clamp(0, 1)
                else:
                    ecin_noisy = ecin_clean

                # Run settling with the noisy input
                # We need to call _settle directly with the noisy input
                zeros_dg  = torch.zeros(B, model.n_dg, device=device)
                zeros_ca3 = torch.zeros(B, model.n_ca3, device=device)
                zeros_ca1 = torch.zeros(B, model.n_ca1, device=device)
                zeros_ec  = torch.zeros(B, model.input_size, device=device)

                _, _, _, ECout_final, _ = model._settle(
                    input_clamp=ecin_noisy,
                    DG_init=zeros_dg,
                    CA3_init=zeros_ca3,
                    CA1_init=zeros_ca1,
                    ECout_init=zeros_ec,
                    ecout_clamp=None,
                    theta=model.THETA_TROUGH,
                )

                pidx = ECout_final.argmax(dim=-1).cpu().numpy()
                pred_correct[:, t] = (pidx == nxt[:, t].numpy())

            acc = float(pred_correct[motif_within].mean())
            results[label].append(acc)

        print(f"    σ={sigma:.2f}  " +
              "  ".join(f"{c[0]:14s}={results[c[0]][-1]:.3f}"
                        for c in conditions))

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Input noise robustness: within-motif prediction accuracy\n"
                 "Gaussian noise (σ) added to the EC_in two-hot clamp.\n"
                 "Higher σ = noisier sensory input.", **STYLE)

    for (label, _, _, _), color in zip(conditions, cond_colors):
        ax.plot(noise_levels, results[label], "o-", lw=2, color=color,
                label=label, markersize=6)

    chance = 1.0 / ALPHABET_SIZE
    ax.axhline(chance, lw=1.5, ls="--", color="red", alpha=0.5,
               label=f"chance ({chance:.2f})")
    ax.set_xlabel("Noise σ (Gaussian, added to EC_in)", **STYLE)
    ax.set_ylabel("Within-motif prediction accuracy", **STYLE)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(noise_levels)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig13_noise_robustness.png"))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Fig 14 – Train-with-lesion, test-intact (lesion during training, restore)
# ══════════════════════════════════════════════════════════════════════════════

def fig_lesion_train_restore(figdir, base_args):
    """
    For each pathway lesion condition:
      1. Train a fresh model WITH the lesion active
      2. At test time, RESTORE the lesioned pathway (turn it back on)
      3. Evaluate within-motif accuracy

    Answers: "if the animal never had [pathway] during learning, can that
    pathway help at retrieval?" Conversely, "what did the model learn without
    [pathway] — is that learning useful when all pathways are available?"

    Compared against:
      - Intact train → intact test (upper bound)
      - Lesion train → lesion test (what lesion-trained model does with lesion ON)
    """
    conditions = [
        ("Intact",       False, False, []),
        ("MSP off",      True,  False, []),
        ("TSP off",      False, True,  []),
        ("Big-loop off", False, False, ["bigloop"]),
    ]
    cond_colors = ["#1a6fc4", "#c44b1a", "#2a8a4a", "#9b5ea0"]

    # Shared eval stream
    seq, prev, nxt, wp, _ = make_embedded_stream(
        batch_size=32, T=400, seed=base_args["seed"] + 3333)
    wp_np = wp.numpy()
    motif_within = ((wp_np >= 0) & (wp_np <= 2))
    device = base_args["device"]

    lesion_dir = os.path.join(figdir, "lesion_train_restore")
    ensure(lesion_dir)

    # Results: for each training condition, store (train_lesion_test, train_intact_test)
    rows = []

    for (label, lm, lt, paths) in conditions:
        tag = label.replace(" ", "_").replace("→", "-")
        ckpt_path = os.path.join(lesion_dir, f"trained_{tag}.pt")

        # Train with lesion
        if not os.path.exists(ckpt_path):
            print(f"\n  Training with {label} …")
            train_model(
                num_epochs      = base_args.get("epochs", 30),
                batch_size      = base_args.get("batch", 64),
                T               = base_args.get("T", 400),
                ckpt_path       = ckpt_path,
                device          = device,
                seed            = base_args["seed"],
                lesion_msp      = lm,
                lesion_tsp      = lt,
                lesion_pathways = paths if paths else None,
                verbose         = False,
            )
        else:
            print(f"\n  {label}: reusing existing {ckpt_path}")

        ck = torch.load(ckpt_path, map_location=device, weights_only=False)

        # ── Eval 1: test WITH lesion still on (same as training) ──────────
        model = build_model(ck.get("config", {}), device)
        restore(model, ck["state_dict"])
        if lm or lt:
            model.set_lesion(msp=lm, tsp=lt)
        for p in paths:
            model.set_pathway(p, on=False)
        model.eval()
        res_lesion = run_stream(model, seq, prev, nxt, capture_layers=False)
        acc_lesion = float(res_lesion["acc"][motif_within].mean())

        # ── Eval 2: test with ALL pathways restored (intact) ──────────────
        model2 = build_model(ck.get("config", {}), device)
        restore(model2, ck["state_dict"])
        model2.reset_lesions()       # everything back on
        model2.eval()
        res_intact = run_stream(model2, seq, prev, nxt, capture_layers=False)
        acc_intact = float(res_intact["acc"][motif_within].mean())

        rows.append(dict(
            label      = label,
            acc_lesion = acc_lesion,
            acc_intact = acc_intact,
        ))
        print(f"    {label:14s}  train+test_lesion={acc_lesion:.3f}  "
              f"train_lesion+test_intact={acc_intact:.3f}")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(
        "Train with lesion → test intact vs test with lesion\n"
        "Can a pathway compensate at test if it was absent during training?\n"
        "Bar pairs: left = test with lesion still on; "
        "right = test with all pathways restored.", **STYLE)

    x = np.arange(len(conditions))
    bar_w = 0.35

    lesion_vals = [r["acc_lesion"] for r in rows]
    intact_vals = [r["acc_intact"] for r in rows]

    ax.bar(x - bar_w/2, lesion_vals, bar_w,
           color=[c for c in cond_colors], edgecolor="black", linewidth=0.5,
           label="test with lesion ON", alpha=0.6, hatch="//")
    ax.bar(x + bar_w/2, intact_vals, bar_w,
           color=[c for c in cond_colors], edgecolor="black", linewidth=0.5,
           label="test with lesion OFF (restored)")

    chance = 1.0 / ALPHABET_SIZE
    ax.axhline(chance, lw=1.5, ls="--", color="red", alpha=0.5,
               label=f"chance ({chance:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Trained\n{c[0]}" for c in conditions],
                       fontsize=10, fontweight="bold")
    ax.set_ylabel("Within-motif prediction accuracy (A→B, B→C, C→D avg)",
                  **STYLE)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="best"); ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig14_lesion_train_restore.png"))

    # Print summary
    print("\n  Summary:")
    print(f"  {'Training':14s}  {'test+lesion':>12s}  {'test+intact':>12s}  "
          f"{'Δ(restore)':>12s}")
    for r in rows:
        delta = r["acc_intact"] - r["acc_lesion"]
        print(f"  {r['label']:14s}  {r['acc_lesion']:>12.3f}  "
              f"{r['acc_intact']:>12.3f}  {delta:>+12.3f}")

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Fig 15 – Confusion matrices: trained with lesion, tested intact
# ══════════════════════════════════════════════════════════════════════════════

def fig_lesion_train_confusion(figdir, base_args):
    """
    For each lesion condition, load the model trained WITH that lesion
    (from fig14's checkpoints), RESTORE all pathways, and show the
    confusion matrix at test time.

    Answers: "what predictions does the model make when it learned without
    pathway X but now has all pathways available?"
    """
    conditions = [
        ("Intact",       False, False, []),
        ("MSP off",      True,  False, []),
        ("TSP off",      False, True,  []),
        ("Big-loop off", False, False, ["bigloop"]),
    ]

    lesion_dir = os.path.join(figdir, "lesion_train_restore")
    device = base_args["device"]

    # Shared test stream
    seq, prev, nxt, wp, _ = make_embedded_stream(
        batch_size=32, T=400, seed=base_args["seed"] + 8888)
    wp_np = wp.numpy()

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        "Confusion matrices: TRAINED with lesion → TESTED intact (all pathways restored)\n"
        "Shows what the model learned to predict when a pathway was absent during training.",
        **STYLE)

    col_labels = WORD_LABELS + ["bg/other"]

    for ax, (label, lm, lt, paths) in zip(axes, conditions):
        tag = label.replace(" ", "_").replace("→", "-")
        ckpt_path = os.path.join(lesion_dir, f"trained_{tag}.pt")

        if not os.path.exists(ckpt_path):
            ax.set_title(f"{label}\n(no checkpoint)", fontsize=10)
            ax.set_visible(False)
            continue

        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(ck.get("config", {}), device)
        restore(model, ck["state_dict"])
        model.reset_lesions()   # ALL pathways restored for testing
        model.eval()

        res = run_stream(model, seq, prev, nxt, capture_layers=False)
        conf = _compute_confusion(res["pred_idx"], wp_np)

        im = ax.imshow(conf, vmin=0, vmax=1, cmap="Reds", aspect="equal")
        ax.set_xticks(range(5)); ax.set_yticks(range(4))
        ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
        ax.set_yticklabels(WORD_LABELS, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted item")
        ax.set_ylabel("Current position" if label == "Intact" else "")
        ax.set_title(f"Trained {label}\n(tested intact)", **STYLE)
        for i in range(4):
            for j in range(5):
                ax.text(j, i, f"{conf[i,j]:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if conf[i,j] > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig15_lesion_train_confusion.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Summary text
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(figdir, results, word_pos, model, decoder_scores=None):
    re = results["pred_err"]; ac = results["acc"]; wp = word_pos
    chance = 1.0 / ALPHABET_SIZE

    lines = ["=" * 64, "HipCat (faithful) analysis summary", "=" * 64, ""]
    lines.append(f"Vocab size: {model.input_size}   chance = {chance:.4f}")
    lines.append(f"Layers: DG={model.n_dg} CA3={model.n_ca3} CA1={model.n_ca1}")
    lines.append("")

    lines.append("Forward-prediction accuracy by transition type:")
    for gid, lbl in zip([-1, 0, 1, 2, 3],
                        ["bg→bg","A→B","B→C","C→D","D→bg"]):
        m = (wp == gid); vals = ac[m]
        if len(vals):
            lines.append(f"  {lbl:8s}  acc={vals.mean():.4f}  "
                         f"err={re[m].mean():.5f}  n={len(vals)}")
    lines.append("")

    if decoder_scores is not None:
        lines.append("Linear decoder (next-item) accuracy per layer:")
        lines.append("  Format: overall  |  mean across word transitions (A→B, B→C, C→D)")
        lines.append("  NOTE: overall is dominated by background (~85% of timesteps,")
        lines.append("        unpredictable by design). Word-transition mean is the real metric.")
        for n in decoder_scores:
            overall = decoder_scores[n]["overall"]
            wordmean = decoder_scores[n]["word_mean"]
            per_pos = decoder_scores[n]["per_pos"]
            lines.append(f"  {n:6s}  overall={overall:.3f}  word_mean={wordmean:.3f}  "
                         f"[A→B={per_pos['A→B']:.2f}  "
                         f"B→C={per_pos['B→C']:.2f}  "
                         f"C→D={per_pos['C→D']:.2f}]")
        lines.append("")

    lines.append("Sparsity check (fraction of units with activity > 0.01):")
    for name, key in [("DG","dg_set"),("CA3","ca3_set"),
                      ("CA1","ca1_set"),("ECout","ecout_set")]:
        if key in results:
            a = results[key]
            frac = float((a > 0.01).mean())
            lines.append(f"  {name:6s}  {frac:.4f}")
    lines.append("")

    lines.append("Diagnostic interpretation:")
    lines.append("  • If acc_BC and acc_CD are well above chance → model learned the chain.")
    lines.append("  • If only acc_AB is high but B→C/C→D are at chance, the model is")
    lines.append("    only learning frequency, not the temporal structure.")
    lines.append("  • If the decoder pulls high CA1 accuracy out of layer activity but the")
    lines.append("    ECout argmax doesn't, the readout (CA1→ECout) is the bottleneck.")
    lines.append("  • If DG/CA3 sparsity is far from target (~1%/~6%), check kWTA k.")

    p = os.path.join(figdir, "summary.txt")
    with open(p, "w") as f: f.write("\n".join(lines) + "\n")
    print(f"  ✓  {p}")
    for l in lines: print("    " + l)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",     default="hipcat_trained.pt")
    ap.add_argument("--figdir",   default="figures")
    ap.add_argument("--device",   default="cpu", choices=["cpu","cuda","auto"])
    ap.add_argument("--batch",    type=int, default=64)
    ap.add_argument("--T",        type=int, default=400)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--rasters_t",type=int, default=80)
    # Lesion sweep options
    ap.add_argument("--run_lesions", action="store_true",
                    help="Train fresh models with each lesion (slow!)")
    ap.add_argument("--lesion_epochs", type=int, default=30)
    ap.add_argument("--lesion_batch",  type=int, default=64)
    ap.add_argument("--lesion_T",      type=int, default=400)
    return ap.parse_args()


def main():
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    ensure(args.figdir)
    print(f"\n[HipCat analysis]  ckpt={args.ckpt}  device={device}  figdir={args.figdir}\n")

    # Load model
    ck    = load_ckpt(args.ckpt, device)
    model = build_model(ck.get("config", {}), device)
    restore(model, ck["state_dict"])
    model.eval()
    print(f"  Model: vocab={model.input_size}  DG={model.n_dg}  "
          f"CA3={model.n_ca3}  CA1={model.n_ca1}")

    # Eval stream
    print("\n[1]  Generating evaluation stream …")
    seq, prev_seq, next_seq, word_pos, _ = make_embedded_stream(
        batch_size=args.batch, T=args.T,
        p_word_onset=0.05, min_gap=6, seed=args.seed)
    print("     Running forward pass (no learning) …")
    results = run_stream(model, seq, prev_seq, next_seq, capture_layers=True)
    wp_np = word_pos.numpy()

    # Figs
    print("\n[2]  Figs …")
    fig_training_curves(ck, args.figdir)
    fig_position_accuracy(results, wp_np, args.figdir)
    fig_onset_aligned(results, wp_np, args.figdir)

    decoder_overall, _ = (fig_decoder_readout(results, wp_np, args.figdir)
                           or ({}, {}))

    fig_overlap_rsm(results, wp_np, args.figdir)
    fig_initial_vs_settled(results, wp_np, args.figdir)
    fig_confusion(results, wp_np, args.figdir)
    fig_rasters(results, wp_np, args.figdir, t_show=args.rasters_t)
    fig_weight_distributions(model, args.figdir)

    # Test-time lesions (always run – fast)
    print("\n[3]  Test-time-only lesion comparison …")
    fig_lesion_test(args.ckpt, args.figdir,
                    base_args=dict(device=device, seed=args.seed))

    # NEW: Confusion matrices per lesion condition
    print("\n[3b] Lesion confusion matrices …")
    fig_lesion_confusion(args.ckpt, args.figdir,
                         base_args=dict(device=device, seed=args.seed))

    # NEW: Gaussian noise robustness
    print("\n[3c] Noise robustness sweep …")
    fig_noise_robustness(args.ckpt, args.figdir,
                         base_args=dict(device=device, seed=args.seed))

    # Training-time lesion sweep (slow – opt-in)
    if args.run_lesions:
        print("\n[4]  Training-time lesion sweep (this is the slow part) …")
        lesion_results = run_lesion_train_sweep(
            args.figdir,
            base_args=dict(device=device, seed=args.seed,
                           epochs=args.lesion_epochs,
                           batch=args.lesion_batch,
                           T=args.lesion_T),
        )
        fig_lesion_train(lesion_results, args.figdir)

    # NEW: Train-with-lesion, test-intact (always with --run_lesions)
    if args.run_lesions:
        print("\n[4b] Train-with-lesion, test-intact (restore pathway at test) …")
        fig_lesion_train_restore(
            args.figdir,
            base_args=dict(device=device, seed=args.seed,
                           epochs=args.lesion_epochs,
                           batch=args.lesion_batch,
                           T=args.lesion_T),
        )

        print("\n[4c] Confusion matrices: trained-with-lesion, tested-intact …")
        fig_lesion_train_confusion(
            args.figdir,
            base_args=dict(device=device, seed=args.seed),
        )

    # Summary
    print("\n[5]  Summary …")
    write_summary(args.figdir, results, wp_np, model,
                  decoder_scores=decoder_overall)

    print(f"\n✓  All figures saved to:  {os.path.abspath(args.figdir)}/\n")


if __name__ == "__main__":
    main()
