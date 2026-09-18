

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model    import HipCatModel
from data     import ALPHABET_SIZE, WORD
from analysis import (load_ckpt, build_model, restore, run_stream,
                      overlap_sim, overlap_rsm, ensure, save, STYLE)
from deviants import make_deviant_stream, DEVIANT_TYPES

WORD_LABELS = ["A", "B", "C", "D"]
# For gap words there is a 5th slot; label maps handle variable length.
POS_COLORS  = ["#e07b39", "#3a9bd5", "#3ac47d", "#9b5ea0", "#c0577f"]


# ══════════════════════════════════════════════════════════════════════════════
# Prototype builders keyed by (standard vs deviant) and word position
# ══════════════════════════════════════════════════════════════════════════════

def _prototypes_from_indices(acts, word_pos, bs, ts, max_pos, n_reps=40):
    """Like _prototypes_by_pos but from explicit (bs, ts) index arrays."""
    B, T, U = acts.shape
    out = np.zeros((max_pos, U))
    wp_sel = word_pos[bs, ts]
    for pos in range(max_pos):
        m = wp_sel == pos
        bb, tt = bs[m], ts[m]
        if len(bb) == 0:
            continue
        n = min(n_reps, len(bb))
        bb, tt = bb[-n:], tt[-n:]
        out[pos] = acts[bb, tt].mean(0)
    return out


def _prototypes_by_pos(acts, word_pos, mask, max_pos, n_reps=40):
    """
    acts     : [B, T, U] settled activity
    word_pos : [B, T]    0..(L-1) within word, -1 bg
    mask     : [B, T] bool  restrict to these timesteps (e.g. standard-only)
    Returns [max_pos, U] mean activity per word position (nan-safe zeros).
    """
    B, T, U = acts.shape
    out = np.zeros((max_pos, U))
    for pos in range(max_pos):
        sel = (word_pos == pos) & mask
        bs, ts = np.where(sel)
        if len(bs) == 0:
            continue
        n = min(n_reps, len(bs))
        bs, ts = bs[-n:], ts[-n:]
        out[pos] = acts[bs, ts].mean(0)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# (1a) Deviant response: novelty of CA3/CA1 at the deviating position
# ══════════════════════════════════════════════════════════════════════════════

def _collect(model, dev_type, batch, T, seed, dev_fraction=0.5, min_gap=8):
    seq, prev, nxt, wp, dp, onset, isd = make_deviant_stream(
        batch_size=batch, T=T, dev_type=dev_type,
        dev_fraction=dev_fraction, min_gap=min_gap, seed=seed)
    res = run_stream(model, seq, prev, nxt, capture_layers=True)
    res["word_pos"] = wp.numpy()
    res["dev_pos"]  = dp.numpy()
    res["is_dev"]   = isd.numpy()
    return res


def fig_dev_response(model, figdir, dev_types, batch, T, seed):
    """
    For each deviant type and each layer (CA3, CA1), measure how much the
    representation at each within-word position DIVERGES from the standard
    prototype for that position.

    novelty(pos) = 1 - overlap( mean_dev_rep(pos), standard_proto(pos) )

    A spike in novelty localised to the deviating position (and, for a
    predictive model, often the position AFTER it) is the oddball signature.
    """
    layers = ["CA3", "CA1"]
    layer_key = {"CA3": "ca3_set", "CA1": "ca1_set"}

    fig, axes = plt.subplots(len(layers), len(dev_types),
                             figsize=(3.6 * len(dev_types), 3.4 * len(layers)),
                             squeeze=False)
    fig.suptitle("Deviant novelty by position  "
                 "(1 − overlap with standard prototype at same position)\n"
                 "Red ✗ marks the deviating position. Higher = more surprised.",
                 **STYLE)

    for ci, dev_type in enumerate(dev_types):
        res = _collect(model, dev_type, batch, T, seed)
        wp, isd, dp = res["word_pos"], res["is_dev"], res["dev_pos"]
        max_pos = int(wp.max()) + 1

        # standard prototypes come from the standard words in the SAME stream
        std_mask = ~isd & (wp >= 0)
        dev_mask =  isd & (wp >= 0)

        # which position is the deviating one for this type (from any dev word)
        dev_positions = np.unique(dp[dp >= 0])

        for ri, layer in enumerate(layers):
            acts = res[layer_key[layer]]
            std_p = _prototypes_by_pos(acts, wp, std_mask, max_pos)
            dev_p = _prototypes_by_pos(acts, wp, dev_mask, max_pos)

            nov = []
            for pos in range(max_pos):
                if std_p[pos].sum() == 0 or dev_p[pos].sum() == 0:
                    nov.append(np.nan)
                else:
                    nov.append(1.0 - overlap_sim(dev_p[pos], std_p[pos]))

            ax = axes[ri][ci]
            xs = np.arange(max_pos)
            bar_colors = [POS_COLORS[p % len(POS_COLORS)] for p in range(max_pos)]
            ax.bar(xs, nov, color=bar_colors, edgecolor="black", linewidth=0.7)
            for dpos in dev_positions:
                if dpos < max_pos:
                    ax.text(dpos, (np.nanmax(nov) if np.isfinite(np.nanmax(nov)) else 1) * 0.9,
                            "✗", ha="center", va="center", color="red",
                            fontsize=16, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(xs)
            labels = [WORD_LABELS[p] if p < 4 else f"+{p-3}" for p in range(max_pos)]
            if dev_type in ("gap", "silent"):
                labels = ["A", "B", "gap", "D"][:max_pos]
            elif dev_type == "ABAD":
                labels = ["A", "B", "A*", "D"][:max_pos]
            ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{layer}\nnovelty", **STYLE)
            if ri == 0:
                ax.set_title(f"deviant = {dev_type}", **STYLE)
            ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig_dev01_novelty_by_position.png"))


# ══════════════════════════════════════════════════════════════════════════════
# (1b) Deviant RSM: standard chain vs deviant chain, CA3 & CA1
# ══════════════════════════════════════════════════════════════════════════════

def fig_dev_rsm(model, figdir, dev_types, batch, T, seed):
    """
    For each deviant type, build overlap RSMs comparing the standard positions
    {A,B,C,D} against the deviant positions of the same word. Rows = standard,
    cols = deviant. On the diagonal, a DROP relative to 1.0 shows the deviant
    representation pulling away from the standard at that position.
    """
    layers = ["CA3", "CA1"]
    layer_key = {"CA3": "ca3_set", "CA1": "ca1_set"}

    for dev_type in dev_types:
        res = _collect(model, dev_type, batch, T, seed)
        wp, isd = res["word_pos"], res["is_dev"]
        max_pos = int(wp.max()) + 1
        std_mask = ~isd & (wp >= 0)
        dev_mask =  isd & (wp >= 0)

        fig, axes = plt.subplots(1, len(layers), figsize=(6.5 * len(layers), 5.5),
                                 squeeze=False)
        fig.suptitle(f"Standard vs deviant representation overlap  "
                     f"(deviant = {dev_type})\n"
                     f"rows = standard position, cols = deviant position; "
                     f"low diagonal = representation diverged", **STYLE)

        if dev_type in ("gap", "silent"):
            pos_labels = ["A", "B", "gap", "D"][:max_pos]
        elif dev_type == "ABAD":
            pos_labels = ["A", "B", "A*", "D"][:max_pos]
        else:
            pos_labels = [WORD_LABELS[p] if p < 4 else f"+{p-3}"
                          for p in range(max_pos)]

        for ax, layer in zip(axes[0], layers):
            acts = res[layer_key[layer]]
            std_p = _prototypes_by_pos(acts, wp, std_mask, max_pos)
            dev_p = _prototypes_by_pos(acts, wp, dev_mask, max_pos)
            M = np.zeros((max_pos, max_pos))
            for i in range(max_pos):
                for j in range(max_pos):
                    if std_p[i].sum() == 0 or dev_p[j].sum() == 0:
                        M[i, j] = np.nan
                    else:
                        M[i, j] = overlap_sim(std_p[i], dev_p[j])
            im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="equal")
            ax.set_xticks(range(max_pos)); ax.set_yticks(range(max_pos))
            ax.set_xticklabels(pos_labels, fontsize=11, fontweight="bold")
            ax.set_yticklabels(pos_labels, fontsize=11, fontweight="bold")
            ax.set_xlabel("deviant position", **STYLE)
            ax.set_ylabel("standard position", **STYLE)
            ax.set_title(layer, **STYLE)
            for i in range(max_pos):
                for j in range(max_pos):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                                fontsize=9,
                                color="white" if M[i, j] < 0.5 else "black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="overlap")

        fig.tight_layout()
        save(fig, os.path.join(figdir, f"fig_dev02_rsm_{dev_type}.png"))


# ══════════════════════════════════════════════════════════════════════════════
# (1c) Cross-condition similarity: standard vs deviant, per position, CA3 & CA1
# ══════════════════════════════════════════════════════════════════════════════

def fig_std_vs_dev_similarity(model, figdir, dev_types, batch, T, seed):
    """
    Direct standard-vs-deviant neural similarity, position by position.

    For each pip position, we compute the similarity between the layer's
    representation during a DEVIANT and its representation during the STANDARD
    at the SAME position. One column per condition; the 'standard' column is the
    reference (standard vs a held-out standard split, so it shows the noise
    ceiling ≈ how self-similar the layer is even with no violation).

    Rows = pip position (A, B, pip-2, D). Columns = standard | each deviant.
    Cell = similarity(dev_rep, std_rep) at that position, 1.0 = identical to
    standard, low = representation pushed away by the violation.

    Two panels: CA3 and CA1. This is the "how similar is the area during a
    deviant vs during standard" heatmap.
    """
    layers = [("CA3", "ca3_set"), ("CA1", "ca1_set")]
    max_pos = 4
    conditions = ["standard"] + [d for d in dev_types if d != "standard"]

    # Collect per-condition prototypes per position, plus a held-out standard
    # prototype set for the reference column.
    proto = {lname: {} for lname, _ in layers}          # proto[layer][cond] = [max_pos, U]
    std_ref = {lname: None for lname, _ in layers}       # held-out standard prototypes

    for cond in conditions:
        res = _collect(model, cond, batch, T, seed,
                       dev_fraction=(0.0 if cond == "standard" else 1.0))
        wp, isd = res["word_pos"], res["is_dev"]
        for lname, key in layers:
            acts = res[key]
            if cond == "standard":
                # split standard occurrences in two halves for a fair
                # standard-vs-standard reference (noise ceiling)
                sel = wp >= 0
                bs, ts = np.where(sel)
                order = np.argsort(ts)  # arbitrary stable ordering
                bs, ts = bs[order], ts[order]
                half = len(bs) // 2
                p_a = _prototypes_from_indices(acts, wp, bs[:half], ts[:half], max_pos)
                p_b = _prototypes_from_indices(acts, wp, bs[half:], ts[half:], max_pos)
                proto[lname]["standard"] = p_a
                std_ref[lname] = p_b
            else:
                dev_mask = isd & (wp >= 0)
                bs, ts = np.where(dev_mask)
                proto[lname][cond] = _prototypes_from_indices(
                    acts, wp, bs, ts, max_pos)

    # Position labels differ per condition (mark the deviating pip)
    def headers(cond):
        return {
            "standard": ["A", "B", "C", "D"],
            "gap":      ["A", "B", "gap", "D"],
            "silent":   ["A", "B", "gap", "D"],
            "AB":       ["A", "B*", "C", "D"],
            "C":        ["A", "B", "C*", "D"],
            "D":        ["A", "B", "C", "D*"],
            "ABAD":     ["A", "B", "A*", "D"],
        }.get(cond, ["p0", "p1", "p2", "p3"])

    fig, axes = plt.subplots(1, 2, figsize=(4.6 * 2 + 1.5, 1.0 * max_pos + 2.6),
                             squeeze=False)
    fig.suptitle("Standard-vs-deviant representational similarity, per position\n"
                 "cell = overlap(deviant rep, standard rep) at that pip.  "
                 "'standard' col = std-vs-std noise ceiling.  Low = pushed away.",
                 **STYLE)

    for ax, (lname, _) in zip(axes[0], layers):
        grid = np.full((max_pos, len(conditions)), np.nan)
        for cj, cond in enumerate(conditions):
            ref = std_ref[lname]
            devp = proto[lname][cond]
            for pos in range(max_pos):
                if devp[pos].sum() == 0 or ref[pos].sum() == 0:
                    continue
                grid[pos, cj] = overlap_sim(devp[pos], ref[pos])

        im = ax.imshow(grid, vmin=0, vmax=1, cmap="magma", aspect="auto")
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, fontweight="bold", rotation=0)
        ax.set_yticks(range(max_pos))
        ax.set_yticklabels(["0", "1", "2", "3"], fontweight="bold")
        ax.set_xlabel("condition", **STYLE)
        if lname == "CA3":
            ax.set_ylabel("pip position", **STYLE)
        ax.set_title(lname, **STYLE)
        for cj, cond in enumerate(conditions):
            hdr = headers(cond)
            for pos in range(max_pos):
                v = grid[pos, cj]
                if np.isfinite(v):
                    ax.text(cj, pos, f"{hdr[pos]}\n{v:.2f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if v < 0.55 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="overlap with standard")

    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig_dev04_std_vs_dev_similarity.png"))


# ══════════════════════════════════════════════════════════════════════════════
# (2) Decode CA1 representations for sequences
# ══════════════════════════════════════════════════════════════════════════════

def _gather_pips(model, dev_types, batch, T, seed):
    """
    Run one stream per sequence type (standard + each deviant) and collect,
    for every within-word pip, the layer activity and the true SOUND index.

    Returns a dict keyed by sequence-type name → dict with:
        ca1  [n, 100], ca3 [n, 80], sound [n] (0..15 true item),
        pos  [n] within-word position, seqtype (str)
    'standard' is always included as the reference sequence.
    """
    seq_types = ["standard"] + [d for d in dev_types if d != "standard"]
    per = {}
    for st in seq_types:
        # dev_fraction=1.0 so a 'deviant' stream is purely that deviant word;
        # 'standard' stream is purely standard words.
        res = _collect(model, st, batch, T, seed,
                       dev_fraction=(0.0 if st == "standard" else 1.0))
        wp  = res["word_pos"]
        base = wp >= 0                         # all within-word pips
        bs, ts = np.where(base)
        per[st] = dict(
            ca1   = res["ca1_set"][bs, ts],
            ca3   = res["ca3_set"][bs, ts],
            sound = res["seq"][bs, ts],
            pos   = wp[bs, ts],
        )
    return per, seq_types


def fig_seq_sound_decode(model, figdir, dev_types, batch, T, seed):
    """
    PER-PIP, PER-SEQUENCE SOUND DECODER for CA1 and CA3.

    A single 16-way logistic-regression decoder (per layer) is trained to read
    out the true SOUND identity from settled layer activity, pooling pips from
    the standard sequence and every deviant sequence. We then report held-out
    test accuracy broken down by (sequence type × pip position), separately for
    CA1 and CA3.

    Reading the figure:
      * Each row = one sequence type (standard, then each deviant).
      * Each column = pip position 0..3 (A, B, pip-2, D — pip-2 is 'gap' for the
        gap sequence, the swapped item for AB/C/D deviants).
      * Cell = P(decoder reads the correct sound at that pip).
      A deviant pip that the decoder still nails means the layer represents the
      ACTUAL sound faithfully even when it violates the chain; a drop means the
      layer's code for that pip is dominated by the expected item instead.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("  [skip sound decode] sklearn not available")
        return

    per, seq_types = _gather_pips(model, dev_types, batch, T, seed)

    # Build pooled training set with a per-example tag (seqtype, pos) so we can
    # split per-cell for test reporting.
    def stack(field):
        return np.concatenate([per[st][field] for st in seq_types], axis=0)

    tags = np.concatenate([
        np.stack([np.full(len(per[st]["pos"]), si), per[st]["pos"]], axis=1)
        for si, st in enumerate(seq_types)
    ], axis=0)   # [N, 2] = (seqtype_index, pos)

    y_all = stack("sound")
    idx_all = np.arange(len(y_all))

    # One shared train/test split across everything, stratified by sound so
    # every class is represented in train.
    tr, te = train_test_split(idx_all, test_size=0.35, random_state=0,
                              stratify=y_all)

    chance = 1.0 / ALPHABET_SIZE
    max_pos = 4
    pos_headers_by_type = {}
    for st in seq_types:
        if st in ("gap", "silent"):
            pos_headers_by_type[st] = ["A", "B", "gap", "D"]
        elif st == "standard":
            pos_headers_by_type[st] = ["A", "B", "C", "D"]
        elif st == "AB":
            pos_headers_by_type[st] = ["A", "B*", "C", "D"]
        elif st == "C":
            pos_headers_by_type[st] = ["A", "B", "C*", "D"]
        elif st == "D":
            pos_headers_by_type[st] = ["A", "B", "C", "D*"]
        elif st == "ABAD":
            pos_headers_by_type[st] = ["A", "B", "A*", "D"]
        else:
            pos_headers_by_type[st] = ["p0", "p1", "p2", "p3"]

    layers = [("CA1", "ca1"), ("CA3", "ca3")]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 1.1 * len(seq_types) + 2.2),
                             squeeze=False)
    fig.suptitle("Per-pip sound decoding for each sequence  (train pooled, test held-out)\n"
                 "cell = P(decode true sound at that pip).  * marks the deviating pip.",
                 **STYLE)

    for ax, (lname, field) in zip(axes[0], layers):
        X = stack(field)
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(X[tr], y_all[tr])
        preds = np.full(len(y_all), -1)
        preds[te] = clf.predict(X[te])

        grid = np.full((len(seq_types), max_pos), np.nan)
        for ri, st in enumerate(seq_types):
            si = seq_types.index(st)
            for pos in range(max_pos):
                cell = (tags[:, 0] == si) & (tags[:, 1] == pos)
                cell_te = cell & np.isin(idx_all, te)
                if cell_te.sum() == 0:
                    continue
                grid[ri, pos] = float(
                    (preds[cell_te] == y_all[cell_te]).mean())

        im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(max_pos))
        ax.set_xticklabels(["0", "1", "2", "3"], fontweight="bold")
        ax.set_yticks(range(len(seq_types)))
        ax.set_yticklabels(seq_types, fontweight="bold")
        ax.set_xlabel("pip position", **STYLE)
        if lname == "CA1":
            ax.set_ylabel("sequence", **STYLE)
        ax.set_title(f"{lname}  (chance={chance:.2f})", **STYLE)
        for ri, st in enumerate(seq_types):
            hdr = pos_headers_by_type[st]
            for pos in range(max_pos):
                v = grid[ri, pos]
                if np.isfinite(v):
                    ax.text(pos, ri, f"{hdr[pos]}\n{v:.2f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if v < 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="decode acc")

    fig.tight_layout()
    save(fig, os.path.join(figdir, "fig_dev03_persequence_sound_decode.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   default="hipcat_trained.pt")
    ap.add_argument("--figdir", default="figures_deviant")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--T",      type=int, default=400)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--dev_types", nargs="*",
                    default=["ABAD"],
                    help=f"any of {DEVIANT_TYPES}  (ABAD = A B A D)")
    ap.add_argument("--restore_intact", action="store_true",
                    help="Turn ALL pathways back ON at test, ignoring the "
                         "checkpoint's saved lesion (trained-lesioned / "
                         "tested-restored).")
    return ap.parse_args()


def main():
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    ensure(args.figdir)
    print(f"\n[HipCat deviant analysis]  ckpt={args.ckpt}  "
          f"dev_types={args.dev_types}  figdir={args.figdir}\n")

    ck    = load_ckpt(args.ckpt, device)
    cfg   = ck.get("config", {})
    model = build_model(cfg, device)
    restore(model, ck["state_dict"])

    # CRITICAL: build_model restores WEIGHTS but not the pathway on/off state.
    # Lesion flags live in model.path_on (runtime), not in state_dict, so we
    # must re-apply them from the saved config or a lesion checkpoint would be
    # analyzed as if intact.
    lm = bool(cfg.get("lesion_msp", False))
    lt = bool(cfg.get("lesion_tsp", False))
    lp = cfg.get("lesion_pathways", []) or []
    if args.restore_intact:
        model.reset_lesions()   # trained-lesioned, tested with ALL pathways ON
    else:
        if lm or lt:
            model.set_lesion(msp=lm, tsp=lt)
        for p in lp:
            model.set_pathway(p, on=False)
    model.eval()

    if args.restore_intact:
        trained = ([f"msp={lm}"] if lm else []) + ([f"tsp={lt}"] if lt else []) + \
                  ([f"pathways={lp}"] if lp else [])
        tstr = " ".join(trained) if trained else "none"
        lesion_str = f"  TRAINED-LESIONED ({tstr}), TESTED RESTORED (all pathways ON)"
    else:
        active_lesions = ([f"msp={lm}"] if lm else []) + \
                         ([f"tsp={lt}"] if lt else []) + \
                         ([f"pathways={lp}"] if lp else [])
        lesion_str = ("  LESIONS: " + " ".join(active_lesions)) if active_lesions \
                     else "  (intact)"
    print(f"  Model: vocab={model.input_size} DG={model.n_dg} "
          f"CA3={model.n_ca3} CA1={model.n_ca1}{lesion_str}\n")

    print("[1] CA3/CA1 novelty by position …")
    fig_dev_response(model, args.figdir, args.dev_types,
                     args.batch, args.T, args.seed)

    print("[2] Standard-vs-deviant RSMs (CA3 & CA1) …")
    fig_dev_rsm(model, args.figdir, args.dev_types,
                args.batch, args.T, args.seed)

    print("[2b] Standard-vs-deviant similarity heatmap (CA3 & CA1) …")
    fig_std_vs_dev_similarity(model, args.figdir, args.dev_types,
                              args.batch, args.T, args.seed)

    print("[3] Per-pip sound decoding (CA1 & CA3) for each sequence …")
    fig_seq_sound_decode(model, args.figdir, args.dev_types,
                         args.batch, args.T, args.seed)

    print(f"\n✓  Deviant figures saved to {os.path.abspath(args.figdir)}/\n")


if __name__ == "__main__":
    main()
