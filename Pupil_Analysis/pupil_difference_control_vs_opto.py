#!/usr/bin/env python3
# ============================================================================
#  DIFFERENCE TRACES — CONTROL vs OPTO ONLY
#
#  Reproduces the two cluster-permutation difference figures from
#  pupil_analysis_main.py, but restricted to the two groups of interest:
#      control  and  opto
#  (control_newsound and opto_late are excluded.)
#
#  Outputs -> outputs_control_vs_opto/
#     C1_difference_control_vs_opto.png              (single panel, all sessions)
#     C2_difference_control_vs_opto_sensitivity.png  (ALL vs low-n excluded)
#     control_vs_opto_cluster_stats.txt              (cluster windows + tests)
#
#  Method is identical to the main script: per session, subtract the normal
#  mean trace from the deviant mean trace, average those difference curves
#  across sessions, and shade time windows where a contiguous cluster survives
#  a 1000-iteration sign-flip permutation test (p < .05).
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------------------------------------------------------
PUPIL_H5 = "EC_opto_pupil.h5"
OUTDIR   = "outputs_control_vs_opto"

KEEP_GROUPS = ["control", "opto"]        # <-- the only change vs the main script

TEST_PHASE_MIN_TRIAL = 101
MIN_DEVIANTS = 10

NA45_ANIMALS       = {"NA04", "NA05"}
NA45_OPTO_DATES    = {"260720", "260721", "260722"}   # light-on  = opto
NA45_CONTROL_DATES = {"260723", "260724", "260725"}   # lights off = control
CONTROL_DATES          = {"260606", "260607", "260717", "260720", "260721", "260722"}
NA03_OPTO_DATES        = {"260606", "260607"}
OPTOLATE_DATES         = {"260618", "260619", "260620"}
CONTROL_NEWSOUND_DATES = {"260710", "260713", "260715"}

PIP_WINDOWS = {"A (0.00-0.25)": (0.00, 0.25), "B (0.25-0.50)": (0.25, 0.50),
               "C (0.50-0.75)": (0.50, 0.75), "D (0.75-1.00)": (0.75, 1.00)}
GROUP_COLORS = {"control": "#7F7F7F", "opto": "#2CA02C"}

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(_HERE, OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)


def resolve_data_file(fname):
    direct = os.path.join(_HERE, fname)
    if os.path.exists(direct):
        return direct
    entries = sorted(os.listdir(_HERE))
    target = fname.lower().replace(" ", "")
    for e in entries:
        if e.lower().replace(" ", "") == target:
            print(f"NOTE: using '{e}' for requested '{fname}'.")
            return os.path.join(_HERE, e)
    h5 = [e for e in entries if e.lower().endswith((".h5", ".hdf5"))]
    raise FileNotFoundError(
        f"Could not find '{fname}' in:\n  {_HERE}\n\n"
        + ("HDF5 files present:\n  " + "\n  ".join(repr(e) for e in h5)
           if h5 else "No .h5 files found here."))


# ----------------------------------------------------------------------------
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
    runs = meta[["animal", "date", "run", "sess"]].drop_duplicates()
    return set(runs.sort_values("run").groupby(["animal", "date"])["sess"].last())


def load_long():
    with pd.HDFStore(resolve_data_file(PUPIL_H5), "r") as store:
        dev = store["/deviant_C"].copy()
        nor = store["/normal"].copy()
    frames = []
    for cond, df in [("deviant", dev), ("normal", nor)]:
        meta = df.index.to_frame(index=False)
        meta["condition"] = cond
        parts = meta["sess"].str.split("_")
        meta["animal"] = parts.str[0]; meta["date"] = parts.str[1]
        meta["run"] = parts.str[2]
        meta["group"] = [classify_session(a, d)
                         for a, d in zip(meta["animal"], meta["date"])]
        vals = df.reset_index(drop=True)
        vals.columns = [float(c) for c in vals.columns]
        frames.append(pd.concat([meta.reset_index(drop=True), vals], axis=1))
    long = pd.concat(frames, ignore_index=True)

    long = long[long["trial"] >= TEST_PHASE_MIN_TRIAL].reset_index(drop=True)
    keep = dedup_runs(long[["animal", "date", "run", "sess"]])
    long = long[long["sess"].isin(keep)].reset_index(drop=True)

    # restrict to the two groups of interest
    n_before = long["sess"].nunique()
    long = long[long["group"].isin(KEEP_GROUPS)].reset_index(drop=True)
    print(f"Restricted to {KEEP_GROUPS}: {n_before} -> "
          f"{long['sess'].nunique()} sessions.")
    return long


def time_axis(long):
    return np.array(sorted(c for c in long.columns if isinstance(c, float)))


def win_max(mat, taxis, w):
    m = (taxis > w[0]) & (taxis <= w[1])
    return np.nanmax(mat[:, m], axis=1)


def sem(a, axis=0):
    a = np.asarray(a, float)
    n = np.sum(~np.isnan(a), axis=axis)
    return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))


def cluster_perm(diff, n_perm=1000, alpha=0.05, seed=1):
    """Sign-flip cluster permutation. Returns (mask, list of (t0,t1,p))."""
    diff = np.asarray(diff, float)
    n_sess, n_t = diff.shape
    mask = np.zeros(n_t, bool)
    if n_sess < 3:
        return mask, []
    rng = np.random.default_rng(seed)
    thr = stats.t.ppf(1 - alpha / 2, df=n_sess - 1)

    def clusters(tv):
        above = np.abs(tv) > thr
        out, i = [], 0
        while i < n_t:
            if above[i]:
                j = i
                while j < n_t and above[j]:
                    j += 1
                out.append((i, j, float(np.sum(tv[i:j])))); i = j
            else:
                i += 1
        return out

    with np.errstate(invalid="ignore"):
        t_obs = np.nan_to_num(np.nanmean(diff, 0) /
                              (np.nanstd(diff, 0, ddof=1) / np.sqrt(n_sess)))
    obs = clusters(t_obs)
    if not obs:
        return mask, []
    null_max = np.empty(n_perm)
    for p in range(n_perm):
        d = diff * rng.choice([-1.0, 1.0], size=n_sess)[:, None]
        with np.errstate(invalid="ignore"):
            tv = np.nan_to_num(np.nanmean(d, 0) /
                               (np.nanstd(d, 0, ddof=1) / np.sqrt(n_sess)))
        null_max[p] = max((abs(m) for _, _, m in clusters(tv)), default=0.0)
    sig = []
    for a, b, m in obs:
        pv = (np.sum(null_max >= abs(m)) + 1) / (n_perm + 1)
        if pv < alpha:
            mask[a:b] = True
            sig.append((a, b, pv))
    return mask, sig


def shade_pips(ax):
    for label, (a, b) in PIP_WINDOWS.items():
        is_C = label.startswith("C")
        ax.axvspan(a, b, color=("#F2C14E" if is_C else "#000000"),
                   alpha=(0.18 if is_C else 0.05), zorder=0)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    y0, y1 = ax.get_ylim()
    for label, (a, b) in PIP_WINDOWS.items():
        ax.text((a + b) / 2, y1 - 0.06 * (y1 - y0), label[0],
                ha="center", va="top", fontsize=9, fontweight="bold")


# ----------------------------------------------------------------------------
def session_means(long, taxis):
    tcols = list(taxis)
    rows = []
    for (sess, grp, cond), sub in long.groupby(["sess", "group", "condition"]):
        mat = sub[tcols].to_numpy(float)
        rows.append({"sess": sess, "group": grp, "condition": cond,
                     "n_trials": len(sub),
                     "mean_trace": np.nanmean(mat, axis=0)})
    return pd.DataFrame(rows)


def paired_diff(sess_df, grp, drop=()):
    piv = {}
    for cond in ["normal", "deviant"]:
        s = sess_df[(sess_df.group == grp) & (sess_df.condition == cond)]
        s = s[~s.sess.isin(drop)]
        piv[cond] = {r.sess: r.mean_trace for _, r in s.iterrows()}
    shared = sorted(set(piv["normal"]) & set(piv["deviant"]))
    if not shared:
        return None, []
    diff = np.vstack([piv["deviant"][s] - piv["normal"][s] for s in shared])
    return diff, shared


def plot_single(sess_df, taxis, lines):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    results = {}
    for grp in KEEP_GROUPS:
        diff, shared = paired_diff(sess_df, grp)
        if diff is None:
            continue
        mu = np.nanmean(diff, 0); se = sem(diff, 0)
        ax.plot(taxis, mu, color=GROUP_COLORS[grp], lw=2,
                label=f"{grp} ({len(shared)} sess)")
        ax.fill_between(taxis, mu - se, mu + se,
                        color=GROUP_COLORS[grp], alpha=0.18, lw=0)
        results[grp] = cluster_perm(diff)
    ax.axhline(0, color="k", lw=0.6); shade_pips(ax)
    y0, y1 = ax.get_ylim(); any_sig = False
    for grp, (mask, sig) in results.items():
        if mask.any():
            any_sig = True
            ax.fill_between(taxis, y0, y1, where=mask,
                            color=GROUP_COLORS[grp], alpha=0.12, lw=0, zorder=0)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("time from pattern onset (s)")
    ax.set_ylabel("Δ pupil (deviant − normal), session-paired")
    sub = ("shaded band = significant cluster (permutation p<.05)" if any_sig
           else "no significant cluster (permutation p<.05)")
    ax.set_title("Deviant − normal: control vs opto (test phase)\n" + sub,
                 fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    f = os.path.join(OUTDIR, "C1_difference_control_vs_opto.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)

    lines.append("\n[ALL SESSIONS] significant clusters")
    for grp, (mask, sig) in results.items():
        if sig:
            for a, b, pv in sig:
                lines.append(f"  {grp:10s}: {taxis[a]:+.2f} to {taxis[b-1]:+.2f} s"
                             f"  (cluster p={pv:.4f})")
        else:
            lines.append(f"  {grp:10s}: none")


def plot_sensitivity(sess_df, taxis, low_n_sess, lines):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, (label, drop) in zip(
            axes, [("ALL sessions", set()),
                   (f">= {MIN_DEVIANTS} deviants", low_n_sess)]):
        results = {}
        for grp in KEEP_GROUPS:
            diff, shared = paired_diff(sess_df, grp, drop=drop)
            if diff is None:
                continue
            mu = np.nanmean(diff, 0); se = sem(diff, 0)
            ax.plot(taxis, mu, color=GROUP_COLORS[grp], lw=2,
                    label=f"{grp} ({len(shared)})")
            ax.fill_between(taxis, mu - se, mu + se,
                            color=GROUP_COLORS[grp], alpha=0.15, lw=0)
            results[grp] = cluster_perm(diff)
        ax.axhline(0, color="k", lw=0.6); shade_pips(ax)
        y0, y1 = ax.get_ylim()
        for grp, (mask, sig) in results.items():
            if mask.any():
                ax.fill_between(taxis, y0, y1, where=mask,
                                color=GROUP_COLORS[grp], alpha=0.12, lw=0,
                                zorder=0)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("time from pattern onset (s)")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        lines.append(f"\n[{label}] significant clusters")
        for grp, (mask, sig) in results.items():
            if sig:
                for a, b, pv in sig:
                    lines.append(f"  {grp:10s}: {taxis[a]:+.2f} to "
                                 f"{taxis[b-1]:+.2f} s  (cluster p={pv:.4f})")
            else:
                lines.append(f"  {grp:10s}: none")
    axes[0].set_ylabel("Δ pupil (deviant − normal), session-paired")
    fig.suptitle("Deviant − normal: control vs opto "
                 "(shaded = significant cluster, permutation p<.05)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f = os.path.join(OUTDIR, "C2_difference_control_vs_opto_sensitivity.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


def main():
    lines = ["=" * 70,
             "DIFFERENCE TRACES — CONTROL vs OPTO ONLY",
             "=" * 70]
    long = load_long()
    taxis = time_axis(long)

    counts = (long[long.condition == "deviant"]
              .groupby(["sess", "group"])["trial"].nunique().reset_index()
              .rename(columns={"trial": "n_deviant"}))
    low_n_sess = set(counts.loc[counts.n_deviant < MIN_DEVIANTS, "sess"])

    for grp in KEEP_GROUPS:
        sub = counts[counts.group == grp]
        lines.append(f"{grp:10s}: {len(sub)} sessions, deviant n "
                     f"{sub.n_deviant.min()}–{sub.n_deviant.max()} "
                     f"(median {int(sub.n_deviant.median())})")
    lines.append(f"low-n sessions (<{MIN_DEVIANTS} deviants): "
                 + (", ".join(sorted(low_n_sess)) if low_n_sess else "none"))

    sess_df = session_means(long, taxis)
    plot_single(sess_df, taxis, lines)
    plot_sensitivity(sess_df, taxis, low_n_sess, lines)

    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(OUTDIR, "control_vs_opto_cluster_stats.txt"), "w") as fh:
        fh.write(txt)
    print("\nDONE. Outputs in", OUTDIR)


if __name__ == "__main__":
    main()
