#!/usr/bin/env python3
# Development version of the pupil analysis.
# Compares normal and deviant sequence responses across session types.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats

PUPIL_H5   = "EC_opto_pupil.h5"
OUTDIR     = "outputs_nolc"
os.makedirs(OUTDIR, exist_ok=True)

CONTROL_DATES          = {"260606", "260607", "260717", "260720", "260721", "260722"}
NA03_OPTO_DATES        = {"260606", "260607"}   # NA03 only: these are opto
OPTOLATE_DATES         = {"260618", "260619", "260620"}
CONTROL_NEWSOUND_DATES = {"260710", "260713", "260715"}
NA45_ANIMALS       = {"NA04", "NA05"}
NA45_OPTO_DATES    = {"260720", "260721", "260722"}   # light-on  = opto
NA45_CONTROL_DATES = {"260723", "260724", "260725"}   # lights off = control
SESSION_TYPE_ORDER = ["control", "opto", "opto_late", "control_newsound"]

PIP_WINDOWS = {
    "A (0.00-0.25)": (0.00, 0.25),
    "B (0.25-0.50)": (0.25, 0.50),
    "C (0.50-0.75)": (0.50, 0.75),   # <- the deviant tone lives here
    "D (0.75-1.00)": (0.75, 1.00),
}
RESPONSE_WINDOW = (0.50, 2.00)

COND_COLORS = {"normal": "#4C72B0", "deviant": "#C44E52"}
STYPE_COLORS = {"control": "#7F7F7F", "opto": "#2CA02C", "opto_late": "#9467BD",
                "control_newsound": "#D62728"}

N_SHUFFLE = 1000
RNG = np.random.default_rng(0)


def classify_session(animal, date):
    if animal in NA45_ANIMALS:
        if date in NA45_CONTROL_DATES:
            return "control"
        return "opto"
    if animal == "NA03" and date in NA03_OPTO_DATES:
        return "opto"
    if date in CONTROL_NEWSOUND_DATES:
        return "control_newsound"
    if date in CONTROL_DATES:
        return "control"
    if date in OPTOLATE_DATES:
        return "opto_late"
    return "opto"


def dedup_runs(meta):
    """When an animal+date has several runs, keep the HIGHEST run number."""
    runs = meta[["animal", "date", "sess"]].drop_duplicates()
    runs["run"] = runs["sess"].str.split("_").str[2]
    return set(runs.sort_values("run").groupby(["animal", "date"])["sess"].last())


def load_long():
    """Return one tidy long DataFrame with trial-level metadata + wide time cols."""
    with pd.HDFStore(PUPIL_H5, "r") as store:
        dev = store["/deviant_C"].copy()
        nor = store["/normal"].copy()
    frames = []
    for cond, df in [("deviant", dev), ("normal", nor)]:
        meta = df.index.to_frame(index=False)          # time, trial, sess
        meta["condition"] = cond
        meta["animal"] = meta["sess"].str.split("_").str[0]
        meta["date"]   = meta["sess"].str.split("_").str[1]
        meta["session_type"] = [classify_session(a, d)
                                for a, d in zip(meta["animal"], meta["date"])]
        vals = df.reset_index(drop=True)
        vals.columns = [float(c) for c in vals.columns]   # numeric time cols
        frames.append(pd.concat([meta.reset_index(drop=True), vals], axis=1))
    long = pd.concat(frames, ignore_index=True)

    keep_sess = dedup_runs(long[["animal", "date", "sess"]])
    dropped = sorted(set(long["sess"].unique()) - keep_sess)
    if dropped:
        print("Dedup: dropping duplicate-day runs ->", dropped)
    long = long[long["sess"].isin(keep_sess)].reset_index(drop=True)
    return long


def time_axis(long):
    return np.array(sorted(c for c in long.columns if isinstance(c, float)))


def win_mean(mat, taxis, w):
    """Mean over a time window (t0,t1] for a trials x time matrix."""
    m = (taxis > w[0]) & (taxis <= w[1])
    return np.nanmean(mat[:, m], axis=1)


def win_max(mat, taxis, w):
    m = (taxis > w[0]) & (taxis <= w[1])
    return np.nanmax(mat[:, m], axis=1)


def shade_pips(ax, ymin=None, ymax=None):
    """Shade the 4 ABCD pip windows and mark the deviant (C) window."""
    for i, (label, (a, b)) in enumerate(PIP_WINDOWS.items()):
        is_C = label.startswith("C")
        ax.axvspan(a, b, color=("#F2C14E" if is_C else "#000000"),
                   alpha=(0.18 if is_C else 0.05), zorder=0)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    y0, y1 = ax.get_ylim()
    ytxt = y1 - 0.06 * (y1 - y0)
    for label, (a, b) in PIP_WINDOWS.items():
        ax.text((a + b) / 2, ytxt, label[0],
                ha="center", va="top", fontsize=9, fontweight="bold")


def sem(a, axis=0):
    a = np.asarray(a, float)
    n = np.sum(~np.isnan(a), axis=axis)
    return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))


def plot_per_session_traces(long, taxis):
    tcols = list(taxis)
    sessions = (long[["animal", "date", "sess", "session_type"]]
                .drop_duplicates()
                .sort_values(["animal", "date"]))
    animals = sorted(long["animal"].unique())
    for animal in animals:
        sess_a = sessions[sessions.animal == animal]
        n = len(sess_a)
        fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.3), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, sess_a.iterrows()):
            for cond in ["normal", "deviant"]:
                sub = long[(long.sess == row.sess) & (long.condition == cond)]
                if sub.empty:
                    continue
                mat = sub[tcols].to_numpy(float)
                mu = np.nanmean(mat, axis=0)
                se = sem(mat, axis=0)
                ax.plot(taxis, mu, color=COND_COLORS[cond], lw=1.6,
                        label=f"{cond} (n={len(sub)})")
                ax.fill_between(taxis, mu - se, mu + se,
                                color=COND_COLORS[cond], alpha=0.25, lw=0)
            ax.set_title(f"{row.date}\n[{row.session_type}]", fontsize=9)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("time from pattern onset (s)")
            shade_pips(ax)
            ax.legend(fontsize=6, loc="upper left", frameon=False)
        axes[0].set_ylabel("z-scored pupil\n(baseline-corrected)")
        fig.suptitle(f"{animal} — pupil: normal vs deviant, per session "
                     f"(shaded = ABCD pips; yellow = deviant C tone)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        f = os.path.join(OUTDIR, f"01_traces_per_session_{animal}.png")
        fig.savefig(f, dpi=140); plt.close(fig)
        print("saved", f)


def session_means(long, taxis):
    """Per (animal,date,sess,session_type,condition) mean trace + scalar summaries."""
    tcols = list(taxis)
    rows = []
    grp = long.groupby(["animal", "date", "sess", "session_type", "condition"])
    for (animal, date, sess, stype, cond), sub in grp:
        mat = sub[tcols].to_numpy(float)
        rec = {"animal": animal, "date": date, "sess": sess,
               "session_type": stype, "condition": cond, "n_trials": len(sub),
               "mean_trace": np.nanmean(mat, axis=0)}
        rec["mean_resp"] = np.nanmean(win_mean(mat, taxis, RESPONSE_WINDOW))
        rec["max_resp"]  = np.nanmean(win_max(mat, taxis, RESPONSE_WINDOW))
        for label, w in PIP_WINDOWS.items():
            rec[f"mean_{label.split()[0]}"] = np.nanmean(win_mean(mat, taxis, w))
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_grandavg_by_type(sess_df, taxis):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, stype in zip(axes, SESSION_TYPE_ORDER):
        for cond in ["normal", "deviant"]:
            sub = sess_df[(sess_df.session_type == stype) &
                          (sess_df.condition == cond)]
            if sub.empty:
                continue
            traces = np.vstack(sub["mean_trace"].to_numpy())
            mu = np.nanmean(traces, axis=0)
            se = sem(traces, axis=0)
            ax.plot(taxis, mu, color=COND_COLORS[cond], lw=2,
                    label=f"{cond} ({len(sub)} sess)")
            ax.fill_between(taxis, mu - se, mu + se,
                            color=COND_COLORS[cond], alpha=0.25, lw=0)
        ax.set_title(stype, fontsize=11)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("time from pattern onset (s)")
        shade_pips(ax)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
    axes[0].set_ylabel("z-scored pupil (session-mean +/- SEM)")
    fig.suptitle("Grand-average pupil by session type — unit = session "
                 "(shaded=ABCD pips, yellow=deviant C tone)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    f = os.path.join(OUTDIR, "02_grandavg_by_session_type.png")
    fig.savefig(f, dpi=140); plt.close(fig)
    print("saved", f)


def plot_difference_traces(sess_df, taxis):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    summary = []
    for stype in SESSION_TYPE_ORDER:
        piv = {}
        for cond in ["normal", "deviant"]:
            s = sess_df[(sess_df.session_type == stype) &
                        (sess_df.condition == cond)][["sess", "mean_trace"]]
            piv[cond] = {r.sess: r.mean_trace for _, r in s.iterrows()}
        shared = sorted(set(piv["normal"]) & set(piv["deviant"]))
        if not shared:
            continue
        diff = np.vstack([piv["deviant"][s] - piv["normal"][s] for s in shared])
        mu = np.nanmean(diff, axis=0)
        se = sem(diff, axis=0)
        ax.plot(taxis, mu, color=STYPE_COLORS[stype], lw=2,
                label=f"{stype} ({len(shared)} sess)")
        ax.fill_between(taxis, mu - se, mu + se,
                        color=STYPE_COLORS[stype], alpha=0.20, lw=0)
        if len(shared) >= 3:
            t, p = stats.ttest_1samp(diff, 0.0, axis=0, nan_policy="omit")
            sig = p < 0.05
            ax.plot(taxis[sig],
                    np.full(sig.sum(), ax.get_ylim()[0]) if False else
                    np.full(sig.sum(), -0.02 - 0.01 * SESSION_TYPE_ORDER.index(stype)),
                    "|", color=STYPE_COLORS[stype], ms=4)
        summary.append((stype, len(shared)))
    ax.axhline(0, color="k", lw=0.6)
    shade_pips(ax)
    ax.set_xlabel("time from pattern onset (s)")
    ax.set_ylabel("Δ pupil (deviant − normal), session-paired")
    ax.set_title("Deviant − normal difference by session type\n"
                 "(ticks below = samples where Δ differs from 0, p<.05 across sessions)",
                 fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    f = os.path.join(OUTDIR, "03_difference_traces_by_type.png")
    fig.savefig(f, dpi=140); plt.close(fig)
    print("saved", f)


def plot_scalar_summaries(sess_df):
    metrics = [("max_resp", "max pupil in resp. window"),
               ("mean_resp", "mean pupil in resp. window")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (metric, ylabel) in zip(axes, metrics):
        positions, ticklabels = [], []
        pos = 0
        for stype in SESSION_TYPE_ORDER:
            for cond in ["normal", "deviant"]:
                vals = sess_df[(sess_df.session_type == stype) &
                               (sess_df.condition == cond)][metric].dropna().values
                pos += 1
                if len(vals):
                    bp = ax.boxplot(vals, positions=[pos], widths=0.6,
                                    patch_artist=True, showfliers=False)
                    for b in bp["boxes"]:
                        b.set(facecolor=COND_COLORS[cond], alpha=0.35)
                    for med in bp["medians"]:
                        med.set(color="k")
                    jitter = pos + RNG.uniform(-0.12, 0.12, len(vals))
                    ax.scatter(jitter, vals, s=18, color=COND_COLORS[cond],
                               edgecolor="k", linewidth=0.3, zorder=3)
                positions.append(pos)
                ticklabels.append(f"{stype}\n{cond}")
            pos += 0.6  # gap between session types
        ax.set_xticks(positions)
        ax.set_xticklabels(ticklabels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(ylabel + f"\nwindow = {RESPONSE_WINDOW[0]}–{RESPONSE_WINDOW[1]} s")
    fig.suptitle("Per-session scalar pupil summaries (each dot = one session)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    f = os.path.join(OUTDIR, "04_scalar_summaries.png")
    fig.savefig(f, dpi=140); plt.close(fig)
    print("saved", f)


def compute_delta_and_shuffle(long, taxis):
    """Per session: observed (deviant-normal) max-in-window difference + shuffle."""
    recs = []
    tcols = list(taxis)
    for sess, sub in long.groupby("sess"):
        stype = sub["session_type"].iloc[0]
        animal = sub["animal"].iloc[0]
        dev = sub[sub.condition == "deviant"][tcols].to_numpy(float)
        nor = sub[sub.condition == "normal"][tcols].to_numpy(float)
        if len(dev) < 3 or len(nor) < 3:
            continue
        obs = (np.nanmean(win_max(dev, taxis, RESPONSE_WINDOW))
               - np.nanmean(win_max(nor, taxis, RESPONSE_WINDOW)))
        pool = np.vstack([dev, nor])
        nd = len(dev)
        null = np.empty(N_SHUFFLE)
        idx = np.arange(len(pool))
        for i in range(N_SHUFFLE):
            RNG.shuffle(idx)
            d = pool[idx[:nd]]; nn = pool[idx[nd:]]
            null[i] = (np.nanmean(win_max(d, taxis, RESPONSE_WINDOW))
                       - np.nanmean(win_max(nn, taxis, RESPONSE_WINDOW)))
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_SHUFFLE + 1)
        recs.append({"sess": sess, "animal": animal, "session_type": stype,
                     "delta_max": obs, "null_lo": np.percentile(null, 2.5),
                     "null_hi": np.percentile(null, 97.5), "p_shuffle": p,
                     "n_dev": len(dev), "n_nor": len(nor)})
    return pd.DataFrame(recs)


def plot_delta(delta_df):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    pos = 0; xt, xl = [], []
    for stype in SESSION_TYPE_ORDER:
        sub = delta_df[delta_df.session_type == stype].sort_values("animal")
        for _, r in sub.iterrows():
            pos += 1
            ax.plot([pos, pos], [r.null_lo, r.null_hi], color="lightgray",
                    lw=6, solid_capstyle="round", zorder=1)
            ax.plot(pos, r.delta_max, "o", color=STYPE_COLORS[stype],
                    ms=7, zorder=3, markeredgecolor="k", markeredgewidth=0.4)
            if r.p_shuffle < 0.05:
                ax.text(pos, r.delta_max, "*", ha="center", va="bottom",
                        fontsize=13, color="k")
            xt.append(pos); xl.append(f"{r.animal}\n{r.sess.split('_')[1]}")
        pos += 0.8
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6, rotation=0)
    ax.set_ylabel("Δ max pupil (deviant − normal)")
    ax.set_title("Per-session deviant−normal Δ with 95% shuffle null (gray)\n"
                 "* = p<.05 vs shuffle;  window "
                 f"{RESPONSE_WINDOW[0]}–{RESPONSE_WINDOW[1]} s", fontsize=11)
    for stype in SESSION_TYPE_ORDER:
        ax.scatter([], [], color=STYPE_COLORS[stype], label=stype)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    f = os.path.join(OUTDIR, "05_delta_pupil_shuffle.png")
    fig.savefig(f, dpi=140); plt.close(fig)
    print("saved", f)


def run_statistics(sess_df, delta_df):
    lines = []
    def W(s=""): lines.append(s)

    W("=" * 78)
    W("STATISTICAL SUMMARY — pupil responses: normal (ABCD) vs deviant (ABAD)")
    W("Unit of analysis = SESSION (per-session means); trials never pooled across")
    W("sessions. Response window = %.2f-%.2f s post pattern onset." % RESPONSE_WINDOW)
    W("=" * 78)

    W("\n[1] Deviant vs Normal within each session type")
    W("    Paired (per session) on max pupil in response window.")
    W("    Wilcoxon signed-rank (nonparametric) + paired t as backup.")
    for stype in SESSION_TYPE_ORDER:
        piv = (sess_df[sess_df.session_type == stype]
               .pivot_table(index="sess", columns="condition", values="max_resp"))
        piv = piv.dropna()
        if len(piv) < 3:
            W(f"  {stype:10s}: n={len(piv)} sessions — too few for a test")
            continue
        d, n = piv["deviant"].values, piv["normal"].values
        try:
            w_stat, w_p = stats.wilcoxon(d, n)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        t_stat, t_p = stats.ttest_rel(d, n)
        W(f"  {stype:10s}: n={len(piv)} sess | "
          f"mean Δ(dev-nor)={np.mean(d-n):+.3f} | "
          f"Wilcoxon p={w_p:.4f} | paired t p={t_p:.4f}")

    W("\n[2] Does the deviant effect (Δ = dev-nor) differ ACROSS session types?")
    W("    One-way ANOVA + Kruskal-Wallis on per-session Δ max pupil.")
    groups = [delta_df[delta_df.session_type == s]["delta_max"].dropna().values
              for s in SESSION_TYPE_ORDER]
    groups_named = list(zip(SESSION_TYPE_ORDER, groups))
    for name, g in groups_named:
        W(f"    {name:10s}: n={len(g)}  mean Δ={np.mean(g):+.3f}  sd={np.std(g,ddof=1) if len(g)>1 else float('nan'):.3f}")
    valid = [g for g in groups if len(g) >= 2]
    if len(valid) >= 2:
        f_stat, f_p = stats.f_oneway(*valid)
        h_stat, h_p = stats.kruskal(*valid)
        W(f"    One-way ANOVA:     F={f_stat:.3f}, p={f_p:.4f}")
        W(f"    Kruskal-Wallis:    H={h_stat:.3f}, p={h_p:.4f}")
    else:
        W("    Not enough groups with data for an omnibus test.")

    W("\n[3] Pairwise session-type comparisons of Δ (Mann-Whitney U, unpaired)")
    import itertools
    for a, b in itertools.combinations(SESSION_TYPE_ORDER, 2):
        ga = delta_df[delta_df.session_type == a]["delta_max"].dropna().values
        gb = delta_df[delta_df.session_type == b]["delta_max"].dropna().values
        if len(ga) >= 2 and len(gb) >= 2:
            u, p = stats.mannwhitneyu(ga, gb, alternative="two-sided")
            W(f"    {a:10s} vs {b:10s}: U={u:.1f}, p={p:.4f} "
              f"(n={len(ga)},{len(gb)})")
        else:
            W(f"    {a:10s} vs {b:10s}: too few sessions")

    W("\n[4] Per-session shuffle test (deviant vs normal, 1000 permutations)")
    for stype in SESSION_TYPE_ORDER:
        sub = delta_df[delta_df.session_type == stype]
        if len(sub):
            nsig = (sub.p_shuffle < 0.05).sum()
            W(f"    {stype:10s}: {nsig}/{len(sub)} sessions significant (p<.05)")

    W("\n[5] Pip-window profile (mean pupil per ABCD pip, session-averaged)")
    for stype in SESSION_TYPE_ORDER:
        W(f"    -- {stype} --")
        for cond in ["normal", "deviant"]:
            sub = sess_df[(sess_df.session_type == stype) & (sess_df.condition == cond)]
            if sub.empty:
                continue
            vals = {p.split()[0]: sub[f"mean_{p.split()[0]}"].mean()
                    for p in PIP_WINDOWS}
            W("       %-8s " % cond + "  ".join(f"{k}={v:+.3f}" for k, v in vals.items()))

    W("\nNotes:")
    W(" - Deviant tone (C) occurs 0.50-0.75 s; expect divergence to build from there.")
    W(" - With only 3 animals, session-level tests are the honest unit; treat p-values")
    W("   as descriptive. A mixed-effects model (pupil ~ condition*session_type +")
    W("   (1|animal)) is the natural next step if you want animal as a random effect.")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUTDIR, "statistics_summary.txt"), "w") as fh:
        fh.write(text)


def main():
    long = load_long()
    taxis = time_axis(long)

    counts = (long.groupby(["session_type", "condition"])
              .agg(n_sessions=("sess", "nunique"), n_trials=("sess", "size")))
    print("Trial / session counts:\n", counts, "\n")
    counts.to_csv(os.path.join(OUTDIR, "session_trial_counts.csv"))

    plot_per_session_traces(long, taxis)
    sess_df = session_means(long, taxis)
    sess_df.drop(columns=["mean_trace"]).to_csv(
        os.path.join(OUTDIR, "per_session_summaries.csv"), index=False)

    plot_grandavg_by_type(sess_df, taxis)
    plot_difference_traces(sess_df, taxis)
    plot_scalar_summaries(sess_df)

    delta_df = compute_delta_and_shuffle(long, taxis)
    delta_df.to_csv(os.path.join(OUTDIR, "per_session_delta_shuffle.csv"), index=False)
    plot_delta(delta_df)

    run_statistics(sess_df, delta_df)
    print("\nDONE. Outputs in", OUTDIR)


if __name__ == "__main__":
    main()