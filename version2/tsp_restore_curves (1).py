"""
tsp_restore_curves.py  –  Training-time TSP ablation with test-time restore.

Question
--------
If the trisynaptic pathway (EC_in->DG, EC_in->CA3, DG->CA3, CA3->CA3, CA3->CA1)
is DISABLED throughout learning, what does the network learn — and does turning
TSP back ON at test time help, hurt, or do nothing?

Three training conditions, each run over multiple seeds:
    intact     : all pathways on during training
    tsp_off    : TSP disabled during training
    (optional) msp_off via --conditions

At every eval epoch, each model is evaluated TWICE on the same frozen weights:
    "test lesioned"  – pathway state identical to training
    "test restored"  – model.reset_lesions(), i.e. all pathways back on

For the intact condition the two evaluations are identical by construction; it
is plotted as the upper-bound reference.

Outputs
-------
    <figdir>/tsp_restore_curves.png   mean +/- SEM accuracy curves over seeds
    <figdir>/tsp_restore_results.npz  raw per-seed arrays for re-plotting

Usage
-----
    python tsp_restore_curves.py --seeds 0 1 2 --epochs 30
    python tsp_restore_curves.py --seeds 0 1 2 3 4 --epochs 30 --eval_every 2
"""

import os
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import HipCatModel
from data  import make_embedded_stream, sound_size, ALPHABET_SIZE


# ── Conditions ────────────────────────────────────────────────────────────────
# name -> dict(msp=bool, tsp=bool, pathways=list)
CONDITIONS = {
    "intact":  dict(msp=False, tsp=False, pathways=[]),
    "tsp_off": dict(msp=False, tsp=True,  pathways=[]),
    "msp_off": dict(msp=True,  tsp=False, pathways=[]),
}

COND_COLORS = {
    "intact":  "#3a3a3a",   # dark grey, matching the project's ablation figures
    "tsp_off": "#c4451a",   # orange-red
    "msp_off": "#1a6fc4",   # blue
}

# TSP learning-rate overrides, populated from CLI in main(). Empty = model defaults.
TSP_LR = {}


def apply_condition(model, cond):
    """Set model pathway state for a given condition dict."""
    model.reset_lesions()
    if cond["msp"] or cond["tsp"]:
        model.set_lesion(msp=cond["msp"], tsp=cond["tsp"])
    for p in cond["pathways"]:
        model.set_pathway(p, on=False)


@torch.no_grad()
def eval_accuracy(model, seq, prev, nxt, wp_np, device):
    """
    Run the model over a frozen eval stream (no learning) and return
    per-transition accuracy dict. Uses eval_step so no weights change.
    """
    T = seq.shape[1]
    accs = []
    for t in range(T):
        cur  = seq[:, t].to(device)
        pv   = prev[:, t].to(device)
        nx   = nxt[:, t].to(device)
        _, pidx, _ = model.eval_step(cur, pv, nx, capture_initial=False)
        accs.append((pidx == nx).float().cpu())
    acc = torch.stack(accs, dim=1).numpy()          # [B, T]

    out = {}
    within = (wp_np >= 0) & (wp_np <= 2)            # A->B, B->C, C->D
    out["within"] = float(acc[within].mean()) if within.any() else np.nan
    for label, pos in [("AB", 0), ("BC", 1), ("CD", 2)]:
        m = (wp_np == pos)
        out[label] = float(acc[m].mean()) if m.any() else np.nan
    bg = (wp_np == -1)
    out["bg"] = float(acc[bg].mean()) if bg.any() else np.nan
    return out


def run_one(cond_name, seed, epochs, eval_every, batch, T, device, verbose=True):
    """
    Train one model under `cond_name` and evaluate at intervals under BOTH
    the lesioned and the restored pathway configuration.

    Returns dict of arrays keyed by (test_mode, metric) with an 'epochs' array.
    """
    cond = CONDITIONS[cond_name]

    model = HipCatModel(input_size=sound_size, device=device, seed=seed,
                        **TSP_LR).to(device)
    apply_condition(model, cond)

    # Frozen eval stream, identical across conditions/seeds for comparability
    e_seq, e_prev, e_nxt, e_wp, _ = make_embedded_stream(
        batch_size=32, T=400, seed=99_000 + seed)
    e_wp_np = e_wp.numpy()

    rec = {"epochs": []}
    for mode in ("lesioned", "restored"):
        for k in ("within", "AB", "BC", "CD", "bg"):
            rec[f"{mode}_{k}"] = []

    def do_eval(ep):
        rec["epochs"].append(ep)
        # 1) test with the SAME pathway state as training
        apply_condition(model, cond)
        model.eval()
        r_les = eval_accuracy(model, e_seq, e_prev, e_nxt, e_wp_np, device)
        # 2) test with ALL pathways restored
        model.reset_lesions()
        model.eval()
        r_res = eval_accuracy(model, e_seq, e_prev, e_nxt, e_wp_np, device)
        # restore training pathway state before continuing to learn
        apply_condition(model, cond)
        for k, v in r_les.items():
            rec[f"lesioned_{k}"].append(v)
        for k, v in r_res.items():
            rec[f"restored_{k}"].append(v)
        return r_les, r_res

    # epoch 0 = before any learning
    do_eval(0)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        if verbose:
            print(f"    [{cond_name} seed{seed}] epoch {ep:3d}/{epochs} "
                  f"training …", end="", flush=True)

        seq, prev, nxt, wp, _ = make_embedded_stream(
            batch_size=batch, T=T, seed=seed * 10_000 + ep)
        seq, prev, nxt = seq.to(device), prev.to(device), nxt.to(device)
        for t in range(T):
            model.train_step(seq[:, t], prev[:, t], nxt[:, t])

        if verbose:
            print(f" done ({time.time() - ep_t0:.1f}s)", flush=True)

        if ep % eval_every == 0 or ep == epochs:
            rl, rr = do_eval(ep)
            if verbose:
                print(f"      ↳ eval ep{ep:3d}  within: "
                      f"lesioned={rl['within']:.3f} restored={rr['within']:.3f}  "
                      f"| B→C les={rl['BC']:.3f}/res={rr['BC']:.3f}  "
                      f"C→D les={rl['CD']:.3f}/res={rr['CD']:.3f}  "
                      f"bg={rl['bg']:.3f}  [total {time.time()-t0:.0f}s]",
                      flush=True)

    return {k: np.array(v, dtype=float) for k, v in rec.items()}


def aggregate(runs, key):
    """Stack per-seed arrays -> (mean, sem) across seeds."""
    M = np.stack([r[key] for r in runs], axis=0)     # [n_seeds, n_evals]
    mean = np.nanmean(M, axis=0)
    n = M.shape[0]
    sem = np.nanstd(M, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, sem


def plot_curves(results, figdir, metrics, chance):
    """
    results: dict cond_name -> list of per-seed record dicts
    Produces one panel per metric; within each panel, per condition,
    solid = test restored, dashed = test lesioned.
    """
    ncol = len(metrics)
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.6), squeeze=False)
    n_seeds = len(next(iter(results.values())))
    fig.suptitle(
        "Training-time ablation with test-time restore  "
        f"(mean ± SEM over {n_seeds} seeds)\n"
        "dashed = tested with training-time pathway state; "
        "solid = tested with ALL pathways restored",
        fontsize=12, fontweight="bold")

    for ax, metric in zip(axes[0], metrics):
        for cond_name, runs in results.items():
            ep = runs[0]["epochs"]
            color = COND_COLORS.get(cond_name, "#555")

            m_res, s_res = aggregate(runs, f"restored_{metric}")
            ax.plot(ep, m_res, lw=2.2, color=color, label=f"{cond_name} → restored")
            ax.fill_between(ep, m_res - s_res, m_res + s_res, color=color, alpha=0.18)

            # For intact, lesioned == restored; skip the duplicate dashed line.
            if cond_name != "intact":
                m_les, s_les = aggregate(runs, f"lesioned_{metric}")
                ax.plot(ep, m_les, lw=2.0, ls="--", color=color,
                        label=f"{cond_name} → lesioned")
                ax.fill_between(ep, m_les - s_les, m_les + s_les,
                                color=color, alpha=0.10)

        ax.axhline(chance, ls=":", color="red", lw=1.4, label=f"chance ({chance:.3f})")
        ax.set_xlabel("Training epoch", fontweight="bold")
        ax.set_ylabel("Forward-prediction accuracy", fontweight="bold")
        title = {"within": "Within-word (A→B, B→C, C→D)",
                 "AB": "A→B", "BC": "B→C", "CD": "C→D",
                 "bg": "Background (control)"}.get(metric, metric)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=7.5, loc="lower right")

    fig.tight_layout()
    path = os.path.join(figdir, "tsp_restore_curves.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ saved → {path}")


def plot_final_bars(results, figdir, chance, seeds, test_mode="restored"):
    """
    Final-epoch accuracy by transition, styled as grouped bars — one bar per
    CONDITION within each transition group (matching the project's existing
    ablation-summary figures).

    test_mode selects which evaluation to plot:
        "restored"  – all pathways switched back on at test (intact eval)
        "lesioned"  – tested with the training-time pathway state
    For the 'intact' condition both are identical.

    Bars that are exactly zero are drawn as a thin visible stub with a "0.00"
    annotation, so a genuine collapse is never confused with a missing run.
    NaN (metric undefined) is annotated "n/a".
    """
    transitions = [("AB", "A→B"), ("BC", "B→C"), ("CD", "C→D"),
                   ("bg", "bg→bg")]
    cond_names = list(results.keys())
    pretty = {"intact": "Intact",
              "msp_off": "MSP off during training",
              "tsp_off": "TSP off during training"}

    n_groups = len(transitions)
    n_cond = len(cond_names)
    bar_w = 0.8 / n_cond
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(2.9 * n_groups + 3, 6.4))
    mode_txt = ("intact eval" if test_mode == "restored"
                else "eval with training-time lesion")
    fig.suptitle(
        f"Final ABCD performance after training-time ablation\n"
        f"({len(seeds)} seed{'s' if len(seeds) != 1 else ''}, {mode_txt}, "
        f"mean ± SEM)",
        fontsize=13, fontweight="bold")

    for ci, cond_name in enumerate(cond_names):
        runs = results[cond_name]
        means, sems, allpts = [], [], []
        for key, _ in transitions:
            vals = np.array([r[f"{test_mode}_{key}"][-1] for r in runs],
                            dtype=float)
            allpts.append(vals)
            finite = vals[np.isfinite(vals)]
            means.append(np.mean(finite) if len(finite) else np.nan)
            sems.append(np.std(finite, ddof=1) / np.sqrt(len(finite))
                        if len(finite) > 1 else 0.0)

        off = (ci - (n_cond - 1) / 2) * bar_w
        color = COND_COLORS.get(cond_name, "#444")
        plot_means = [0.0 if not np.isfinite(m) else m for m in means]
        ax.bar(x + off, plot_means, bar_w * 0.9,
               yerr=[0 if not np.isfinite(s) else s for s in sems],
               capsize=3, color=color, edgecolor="black", linewidth=0.8,
               label=pretty.get(cond_name, cond_name))

        # annotate zero / NaN so absence is never ambiguous
        for gi, m in enumerate(means):
            if not np.isfinite(m):
                ax.text(x[gi] + off, 0.02, "n/a", ha="center", va="bottom",
                        fontsize=8, rotation=90, color="black")
            elif m < 1e-6:
                ax.plot([x[gi] + off - bar_w * 0.45, x[gi] + off + bar_w * 0.45],
                        [0, 0], color=color, lw=3, solid_capstyle="butt")
                ax.text(x[gi] + off, 0.02, "0.00", ha="center", va="bottom",
                        fontsize=8, rotation=90, color="black")

        # individual seed dots (only when >1 seed, else they clutter)
        if len(seeds) > 1:
            for gi, vals in enumerate(allpts):
                v = vals[np.isfinite(vals)]
                if len(v) == 0:
                    continue
                jit = (np.random.RandomState(gi + ci).rand(len(v)) - 0.5) * bar_w * 0.35
                ax.scatter(np.full(len(v), x[gi] + off) + jit, v,
                           s=15, color="black", zorder=5, alpha=0.7, linewidths=0)

    ax.axhline(chance, ls="--", color="red", lw=1.6,
               label=f"chance ({chance:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in transitions], fontsize=11)
    ax.set_ylabel(f"Final accuracy ({mode_txt}, mean ± SEM)", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    order = list(range(1, len(handles))) + [0]      # chance line last
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              fontsize=9, loc="upper right")

    fig.tight_layout()
    path = os.path.join(figdir, f"tsp_restore_final_bars_{test_mode}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved → {path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--figdir", type=str, default="figures_tsp_restore")
    ap.add_argument("--conditions", nargs="+", default=["intact", "tsp_off"],
                    choices=list(CONDITIONS.keys()))
    ap.add_argument("--metrics", nargs="+",
                    default=["within", "BC", "CD", "bg"],
                    choices=["within", "AB", "BC", "CD", "bg"])
    # TSP learning-rate overrides (default: model's tuned 0.05/0.05/0.05/0.02).
    # Pass Schapiro-faithful values with --lr_ecin_dg 0.2 --lr_ecin_ca3 0.2
    # --lr_ca3_ca3 0.2 --lr_ca3_ca1 0.05, or use --lr_tsp_scale.
    ap.add_argument("--lr_ecin_dg",  type=float, default=None)
    ap.add_argument("--lr_ecin_ca3", type=float, default=None)
    ap.add_argument("--lr_ca3_ca3",  type=float, default=None)
    ap.add_argument("--lr_ca3_ca1",  type=float, default=None)
    ap.add_argument("--lr_tsp_scale", type=float, default=None,
                    help="Multiply ALL TSP lrs by this factor on top of model "
                         "defaults (e.g. 4 → ~0.2/0.2/0.2/0.08).")
    return ap.parse_args()


def main():
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    os.makedirs(args.figdir, exist_ok=True)

    # Resolve TSP learning-rate overrides into the module-level store read by run_one.
    lr_dg, lr_c3, lr_r, lr_c1 = (args.lr_ecin_dg, args.lr_ecin_ca3,
                                 args.lr_ca3_ca3, args.lr_ca3_ca1)
    if args.lr_tsp_scale is not None:
        s = args.lr_tsp_scale
        if lr_dg is None: lr_dg = 0.05 * s
        if lr_c3 is None: lr_c3 = 0.05 * s
        if lr_r  is None: lr_r  = 0.05 * s
        if lr_c1 is None: lr_c1 = 0.02 * s
    for k, v in [("lr_ecin_dg", lr_dg), ("lr_ecin_ca3", lr_c3),
                 ("lr_ca3_ca3", lr_r), ("lr_ca3_ca1", lr_c1)]:
        if v is not None:
            TSP_LR[k] = v
    if TSP_LR:
        print(f"  TSP learning-rate overrides: {TSP_LR}")

    print(f"\n[TSP train-ablation / test-restore]  device={device}  "
          f"seeds={args.seeds}  epochs={args.epochs}  "
          f"conditions={args.conditions}\n")

    results = {}
    total_runs = len(args.conditions) * len(args.seeds)
    run_i = 0
    for cond_name in args.conditions:
        print(f"\n  === condition: {cond_name} ===", flush=True)
        runs = []
        for seed in args.seeds:
            run_i += 1
            print(f"  --- run {run_i}/{total_runs}: {cond_name}, seed {seed} ---",
                  flush=True)
            runs.append(run_one(cond_name, seed, args.epochs, args.eval_every,
                                args.batch, args.T, device))
        results[cond_name] = runs

    # Save raw arrays for re-plotting without retraining
    npz = {}
    for cond_name, runs in results.items():
        for i, r in enumerate(runs):
            for k, v in r.items():
                npz[f"{cond_name}/seed{args.seeds[i]}/{k}"] = v
    np.savez(os.path.join(args.figdir, "tsp_restore_results.npz"), **npz)
    print(f"\n  ✓ saved → {os.path.join(args.figdir, 'tsp_restore_results.npz')}")

    plot_curves(results, args.figdir, args.metrics, 1.0 / ALPHABET_SIZE)
    plot_final_bars(results, args.figdir, 1.0 / ALPHABET_SIZE, args.seeds,
                    test_mode="restored")
    plot_final_bars(results, args.figdir, 1.0 / ALPHABET_SIZE, args.seeds,
                    test_mode="lesioned")

    # Console summary at final epoch
    print("\n  Final-epoch within-word accuracy (mean ± SEM over seeds):")
    for cond_name, runs in results.items():
        m, s = aggregate(runs, "restored_within")
        ml, sl = aggregate(runs, "lesioned_within")
        print(f"    {cond_name:9s}  lesioned-test={ml[-1]:.3f}±{sl[-1]:.3f}   "
              f"restored-test={m[-1]:.3f}±{s[-1]:.3f}")
    print()


if __name__ == "__main__":
    main()