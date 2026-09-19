#!/usr/bin/env python3
# Within-session hit-rate analysis split by normal and deviant trials.
# Checks whether performance changes are general or condition-specific.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import pupil_behaviour_control_opto as B

N_BINS = 10
COND_STYLE = {"normal": dict(ls="-", marker="o"),
              "deviant": dict(ls="--", marker="s")}
OUTDIR = B.OUTDIR


def sem(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan


def main():
    trials = B.load_trials()
    groups = [g for g in B.GROUP_ORDER if g in set(trials["group"])]

    trials = trials.copy()
    trials["rank"] = trials.groupby("sess_key")["trial_num"].rank(pct=True)
    bins = np.linspace(0, 1, N_BINS + 1)
    trials["bin"] = pd.cut(trials["rank"], bins, include_lowest=True)
    centers = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(1, len(groups), figsize=(5.6 * len(groups), 4.4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    lines = ["=" * 70,
             "WITHIN-SESSION HIT RATE by CONDITION (control vs opto)",
             "=" * 70,
             "Question: is the end-of-session performance drop general (normal",
             "AND deviant) or selective to deviant trials?",
             f"Bins: {N_BINS} equal slices of within-session position (test phase)."]

    for ax, g in zip(axes, groups):
        for cond in ["normal", "deviant"]:
            sub = trials[(trials.group == g) & (trials.condition == cond)]
            per_sess = sub.groupby(["sess_key", "bin"], observed=True)["hit"].mean()
            m = per_sess.groupby("bin", observed=True).mean()
            e = per_sess.groupby("bin", observed=True).apply(lambda x: sem(x.values))
            mvals = [m.get(b, np.nan) for b in m.index.categories]
            evals = [e.get(b, np.nan) for b in e.index.categories]
            ax.errorbar(centers, mvals, yerr=evals,
                        color=B.GROUP_COLORS[g], lw=1.8, capsize=2,
                        label=cond, **COND_STYLE[cond])
            s2 = sub.dropna(subset=["hit"])
            if s2["rank"].notna().sum() > 10:
                r, p = stats.pointbiserialr(s2["hit"], s2["rank"])
                lines.append(f"  {g:8s} {cond:8s}: hit-vs-position "
                             f"point-biserial r={r:+.3f} p={p:.4g} "
                             f"(negative = decline over session)")
        ax.set_title(g, fontsize=12, color=B.GROUP_COLORS[g])
        ax.set_xlabel("relative position within session (0 = start, 1 = end)")
        ax.set_ylim(0, 1.02)
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.legend(fontsize=9, frameon=False, title="trial type")
    axes[0].set_ylabel("hit rate (P correct)")
    fig.suptitle("Within-session hit rate, normal vs deviant\n"
                 "parallel decline = general session effect, not deviant-selective",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    f = os.path.join(OUTDIR, "B5b_session_timecourse_by_condition.png")
    fig.savefig(f, dpi=150); plt.close(fig); print("saved", f)

    lines.append("\nIs the within-session decline SELECTIVE to deviant trials?")
    lines.append("Per-session slope of hit rate vs position, deviant vs normal")
    lines.append("(paired within session; a difference = condition-selective drop):")
    for g in groups:
        sub = trials[trials.group == g]
        slopes = {}
        for cond in ["normal", "deviant"]:
            sc = sub[sub.condition == cond]
            per = []
            for sk, gg in sc.groupby("sess_key"):
                gg = gg.dropna(subset=["hit", "rank"])
                if gg["rank"].nunique() >= 3:
                    b, _, _, _, _ = stats.linregress(gg["rank"], gg["hit"])
                    per.append((sk, b))
            slopes[cond] = pd.Series(dict(per))
        common = slopes["normal"].index.intersection(slopes["deviant"].index)
        if len(common) >= 3:
            dv = slopes["deviant"].loc[common].values
            nv = slopes["normal"].loc[common].values
            try:
                w, p = stats.wilcoxon(dv, nv)
            except ValueError:
                p = np.nan
            lines.append(f"  {g:8s}: n={len(common)} sess | "
                         f"mean slope normal={nv.mean():+.3f} "
                         f"deviant={dv.mean():+.3f} | "
                         f"deviant-vs-normal Wilcoxon p={p:.4f}")
            lines.append(f"           (p>.05 => decline NOT steeper for deviant "
                         f"= general, not selective)")
        else:
            lines.append(f"  {g:8s}: too few paired sessions")

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUTDIR, "within_session_hitrate_stats.txt"), "w") as fh:
        fh.write(txt)
    print("DONE. Outputs in", OUTDIR)


if __name__ == "__main__":
    main()
