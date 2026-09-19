#!/usr/bin/env python3
# ============================================================================
#  COMBINED pupil analysis — v1 (session-level) + v2 (sensitivity + mixed model)
#  UPDATED for the expanded dataset (EC_opto_pupil2.h5): 5 animals, 14 dates.
#  *** TEST PHASE ONLY *** (trial >= 101).
#
#  NORMAL (standard 4-tone pattern) vs DEVIANT (violated pattern) pupil responses.
#
#  ---------------------------------------------------------------------------
#  GROUPING RULES (per user instruction)
#  ---------------------------------------------------------------------------
#   * NA04 and NA05 are ALL OPTO (they only appear on 260720/260721 and run the
#     ORIGINAL sound set -> a fresh cohort on the original protocol).
#   * For NA01/NA02/NA03 only:
#        260606, 260607, 260717, 260720, 260721 -> 'control'   (MERGED)
#        260710, 260713, 260715                 -> 'control_newsound'
#        260618, 260619, 260620                 -> 'opto_late'
#        everything else (260603/604/605)       -> 'opto'
#   * NA03 exception: 260606 and 260607 are OPTO for NA03 only. This does NOT
#     apply to the merged-in July control dates.
#
#   NOTE ON THE MERGE: 260717/260720/260721 use the NEW tone set
#   (normal 10;13;13;10 / deviant 10;13;16;19) while 260606/260607 use the
#   ORIGINAL set (normal 10;15;20;25 / deviant 10;15;10;25). They are merged
#   into one 'control' group per instruction, so this group spans two sound
#   sets -- worth stating in any write-up.
#
#  Verified against the metadata (EC_opto_td_df2.h5), the sound sets are:
#     original : normal 10;15;20;25   deviant 10;15;10;25
#     new      : normal 10;13;13;10   deviant 10;13;16;19  (260717/720/721)
#                normal 10;13;13;10   deviant 10;15;20;25  (260713/715)
#     NA01-03 260710 still uses the ORIGINAL set but is grouped with
#     control_newsound per instruction (it is the first of the new block).
#     NA04/NA05 use the ORIGINAL set.
#  control_newsound (260710/713/715) is kept SEPARATE because its deviant is
#  still the old pattern while its normal has changed.
#
#  ---------------------------------------------------------------------------
#  RUN DEDUP: if an animal+date has more than one run, keep the HIGHEST run
#  number (e.g. _001 over _000). This handles NA02_260619 and NA03_260606, and
#  also NA04_260721 which is run _002.
#  ---------------------------------------------------------------------------
#
#  Data model: /normal and /deviant_C frame_tables, one row per trial, 400 time
#  columns -1.00..+2.99 s at 100 Hz, already z-scored and baseline-corrected.
#  sess = "<animal>_<date>_<run>".
#  Pips (250 ms each): A 0-0.25 | B 0.25-0.5 | C 0.5-0.75 (the deviant tone) |
#  D 0.75-1.0 s. Scalar response window: 0.5-2.0 s.
#
#  PLOTS PRODUCED (all v1 + all v2):
#    00  per-session deviant-count table
#    01  per-session traces, one figure per animal
#    02b plain grand-average by group          (v1)
#    02  grand-average ALL vs low-n-excluded   (v2 sensitivity)
#    03b single-panel deviant-normal difference, cluster-shaded (v1)
#    03  two-panel difference ALL vs excluded  (v2 sensitivity)
#    04  scalar summaries (max/mean in window)
#    05  per-session delta vs 1000-shuffle null
#  STATS: paired Wilcoxon/t per group; ANOVA + Kruskal-Wallis + pairwise
#    Mann-Whitney across groups; per-session permutation test; pip profile;
#    all run twice (ALL sessions vs low-n excluded); plus a trial-level linear
#    mixed model max_resp ~ condition*group + (1|animal) + session variance.
# ============================================================================

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import statsmodels.formula.api as smf

# ----------------------------------------------------------------------------
# Config  (bare filenames: data files sit in the same folder as this script)
# ----------------------------------------------------------------------------
PUPIL_H5 = "EC_opto_pupil.h5"
OUTDIR   = "outputs_main"


TEST_PHASE_MIN_TRIAL = 101       # test phase = block 3 = trial >= 101
MIN_DEVIANTS = 10                # low-n flag / sensitivity threshold

# NA04/NA05 are a later cohort: light-on (opto) days vs lights-off (control) days.
NA45_ANIMALS      = {"NA04", "NA05"}
NA45_OPTO_DATES   = {"260720", "260721", "260722"}   # light-on  = opto
NA45_CONTROL_DATES= {"260723", "260724", "260725"}   # lights off = control
# NOTE: 260717/260720/260721 are merged into CONTROL per user instruction.
# They use the NEW tone set (normal 10;13;13;10 / deviant 10;13;16;19) while
# 260606/260607 use the ORIGINAL set, so the merged control spans two sound
# sets. 260710/260713/260715 remain separate as control_newsound.
CONTROL_DATES          = {"260606", "260607", "260717", "260720", "260721", "260722"}
NA03_OPTO_DATES        = {"260606", "260607"}   # NA03 only: these are opto
OPTOLATE_DATES         = {"260618", "260619", "260620"}
CONTROL_NEWSOUND_DATES = {"260710", "260713", "260715"}

GROUP_ORDER = ["control", "opto", "opto_late", "control_newsound"]

PIP_WINDOWS = {
    "A (0.00-0.25)": (0.00, 0.25),
    "B (0.25-0.50)": (0.25, 0.50),
    "C (0.50-0.75)": (0.50, 0.75),   # the deviant tone
    "D (0.75-1.00)": (0.75, 1.00),
}
RESPONSE_WINDOW = (0.50, 2.00)

COND_COLORS = {"normal": "#4C72B0", "deviant": "#C44E52"}
GROUP_COLORS = {"control": "#7F7F7F", "opto": "#2CA02C", "opto_late": "#9467BD", "control_newsound": "#D62728"}

# Separate highlight colour for statistically significant time clusters.
# This keeps significance distinct from pip shading and group trace colours.
SIGNIF_COLOR = "#E78AC3"
SIGNIF_ALPHA = 0.28

N_SHUFFLE = 1000
RNG = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# Robust file location: resolve data files relative to THIS script's folder,
# with a case-insensitive fallback and a helpful listing if not found.
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(_HERE, OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)


def resolve_data_file(fname):
    direct = os.path.join(_HERE, fname)
    if os.path.exists(direct):
        return direct
    try:
        entries = sorted(os.listdir(_HERE))
    except Exception:
        entries = []
    target = fname.lower().replace(" ", "")
    for e in entries:
        if e.lower().replace(" ", "") == target:
            print(f"NOTE: using '{e}' for requested '{fname}'.")
            return os.path.join(_HERE, e)
    h5 = [e for e in entries if e.lower().endswith((".h5", ".hdf5"))]
    msg = [f"Could not find '{fname}' in:", f"  {_HERE}", ""]
    if h5:
        msg.append("HDF5-like files that ARE present:")
        msg += [f"  {repr(e)}" for e in h5]
        msg.append("")
        msg.append("Check the exact spelling above (repr reveals hidden spaces).")
    else:
        msg.append("No .h5/.hdf5 files found in this folder.")
        msg.append("Folder contents:")
        msg += [f"  {repr(e)}" for e in entries[:40]]
    raise FileNotFoundError("\n".join(msg))




# ----------------------------------------------------------------------------
# Load & tidy
# ----------------------------------------------------------------------------
def classify_session(animal, date):
    """Assign a session to an analysis group. See header for the rules."""
    if animal in NA45_ANIMALS:
        if date in NA45_CONTROL_DATES:
            return "control"               # NA04/NA05 lights-off days
        return "opto"                      # NA04/NA05 light-on days
    # --- NA01 / NA02 / NA03 below ---
    if animal == "NA03" and date in NA03_OPTO_DATES:
        return "opto"                      # NA03 260606/260607 are opto
    if date in CONTROL_NEWSOUND_DATES:
        return "control_newsound"
    if date in CONTROL_DATES:
        return "control"
    if date in OPTOLATE_DATES:
        return "opto_late"
    return "opto"


def dedup_runs(meta):
    """If an animal+date has several runs, keep the HIGHEST run number."""
    runs = meta[["animal", "date", "run", "sess"]].drop_duplicates()
    keep = runs.sort_values("run").groupby(["animal", "date"])["sess"].last()
    return set(keep)


def load_long():
    with pd.HDFStore(resolve_data_file(PUPIL_H5), "r") as store:
        dev = store["/deviant_C"].copy()
        nor = store["/normal"].copy()
    frames = []
    for cond, df in [("deviant", dev), ("normal", nor)]:
        meta = df.index.to_frame(index=False)          # time, trial, sess
        meta["condition"] = cond
        parts = meta["sess"].str.split("_")
        meta["animal"] = parts.str[0]
        meta["date"]   = parts.str[1]
        meta["run"]    = parts.str[2]
        meta["group"]  = [classify_session(a, d)
                          for a, d in zip(meta["animal"], meta["date"])]
        vals = df.reset_index(drop=True)
        vals.columns = [float(c) for c in vals.columns]
        frames.append(pd.concat([meta.reset_index(drop=True), vals], axis=1))
    long = pd.concat(frames, ignore_index=True)

    # ---- test-phase filter (explicit + reported) ----
    n0 = len(long)
    long = long[long["trial"] >= TEST_PHASE_MIN_TRIAL].reset_index(drop=True)
    print(f"Test-phase filter (trial >= {TEST_PHASE_MIN_TRIAL}): "
          f"removed {n0 - len(long)} trials, kept {len(long)}.")

    # ---- run dedup (keep highest run per animal+date) ----
    keep_sess = dedup_runs(long[["animal", "date", "run", "sess"]])
    dropped = sorted(set(long["sess"].unique()) - keep_sess)
    if dropped:
        print("Dedup (kept highest run) -> dropped:", dropped)
    long = long[long["sess"].isin(keep_sess)].reset_index(drop=True)
    return long


def time_axis(long):
    return np.array(sorted(c for c in long.columns if isinstance(c, float)))


def win_mean(mat, taxis, w):
    m = (taxis > w[0]) & (taxis <= w[1])
    return np.nanmean(mat[:, m], axis=1)


def win_max(mat, taxis, w):
    m = (taxis > w[0]) & (taxis <= w[1])
    return np.nanmax(mat[:, m], axis=1)


def sem(a, axis=0):
    a = np.asarray(a, float)
    n = np.sum(~np.isnan(a), axis=axis)
    return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))


def cluster_perm_sig(diff, taxis, n_perm=1000, alpha=0.05, seed=1):
    """Cluster-based permutation (sign-flip) on a sessions x time matrix.
    Returns a boolean time-mask of significant, contiguous clusters."""
    diff = np.asarray(diff, float)
    n_sess, n_t = diff.shape
    sig = np.zeros(n_t, dtype=bool)
    if n_sess < 3:
        return sig
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
        return sig
    null_max = np.empty(n_perm)
    for p in range(n_perm):
        d = diff * rng.choice([-1.0, 1.0], size=n_sess)[:, None]
        with np.errstate(invalid="ignore"):
            tv = np.nan_to_num(np.nanmean(d, 0) /
                               (np.nanstd(d, 0, ddof=1) / np.sqrt(n_sess)))
        cl = clusters(tv)
        null_max[p] = max((abs(m) for _, _, m in cl), default=0.0)
    for a, b, mass in obs:
        if (np.sum(null_max >= abs(mass)) + 1) / (n_perm + 1) < alpha:
            sig[a:b] = True
    return sig


# Which pips are the DEVIANT tone(s), per group. For the original sound set the
# deviant differs from the standard only at C (0.50-0.75 s). For control_newsound
# the standard is 10;13;13;10 and the deviant is 10;15;20;25, which differs at
# B, C AND D (only A matches) -> highlight B/C/D. group=None -> C only.
DEVIANT_PIPS = {"control_newsound": {"B", "C", "D"}}


def deviant_pips_for(group):
    return DEVIANT_PIPS.get(group, {"C"})


def shade_pips(ax, group=None):
    dev = deviant_pips_for(group)
    for label, (a, b) in PIP_WINDOWS.items():
        is_dev = label.split()[0] in dev
        ax.axvspan(a, b, color=("#F2C14E" if is_dev else "#000000"),
                   alpha=(0.18 if is_dev else 0.05), zorder=0)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    y0, y1 = ax.get_ylim()
    for label, (a, b) in PIP_WINDOWS.items():
        ax.text((a + b) / 2, y1 - 0.06 * (y1 - y0), label[0],
                ha="center", va="top", fontsize=8, fontweight="bold")


def groups_present(df):
    return [g for g in GROUP_ORDER if g in set(df["group"])]


# ============================================================================
# 00 — per-session count table
# ============================================================================
def build_count_table(long):
    g = (long.groupby(["sess", "animal", "date", "group", "condition"])
             ["trial"].nunique().unstack("condition").fillna(0).astype(int)
             .rename(columns={"deviant": "n_deviant", "normal": "n_normal"})
             .reset_index())
    for c in ["n_deviant", "n_normal"]:
        if c not in g:
            g[c] = 0
    g["ratio_normal_per_deviant"] = (g["n_normal"] /
                                     g["n_deviant"].replace(0, np.nan)).round(1)
    g["deviant_below_min"] = g["n_deviant"] < MIN_DEVIANTS
    g["group"] = pd.Categorical(g["group"], categories=GROUP_ORDER, ordered=True)
    return g.sort_values(["group", "n_deviant"])


def plot_count_table(counts):
    order = counts.sort_values(["group", "n_deviant"])
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(order))))
    ypos = np.arange(len(order))
    ax.barh(ypos, order["n_deviant"],
            color=[GROUP_COLORS[g] for g in order["group"]],
            edgecolor="k", linewidth=0.4)
    for y, (_, r) in zip(ypos, order.iterrows()):
        ax.text(r["n_deviant"] + 0.4, y, f"{r['n_deviant']}", va="center", fontsize=7)
    ax.axvline(MIN_DEVIANTS, color="red", ls="--", lw=1.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r.animal}_{r.date} [{r.group}]"
                        for _, r in order.iterrows()], fontsize=6.5)
    ax.set_xlabel("number of DEVIANT trials in session")
    ax.set_title("Per-session deviant-trial counts\n"
                 f"(red line = low-n threshold, {MIN_DEVIANTS}; "
                 "left of it = excluded in the sensitivity analysis)", fontsize=11)
    handles = [Patch(facecolor="red", label=f"cut = {MIN_DEVIANTS}")]
    handles += [Patch(facecolor=GROUP_COLORS[g], edgecolor="k", label=g)
                for g in groups_present(counts)]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    f = os.path.join(OUTDIR, "00_deviant_counts_per_session.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


# ============================================================================
# per-session mean traces + scalar summaries
# ============================================================================
def session_means(long, taxis):
    tcols = list(taxis)
    rows = []
    for (animal, date, sess, grp, cond), sub in long.groupby(
            ["animal", "date", "sess", "group", "condition"]):
        mat = sub[tcols].to_numpy(float)
        rec = {"animal": animal, "date": date, "sess": sess, "group": grp,
               "condition": cond, "n_trials": len(sub),
               "mean_trace": np.nanmean(mat, axis=0),
               "mean_resp": np.nanmean(win_mean(mat, taxis, RESPONSE_WINDOW)),
               "max_resp": np.nanmean(win_max(mat, taxis, RESPONSE_WINDOW))}
        for label, w in PIP_WINDOWS.items():
            rec[f"mean_{label.split()[0]}"] = np.nanmean(win_mean(mat, taxis, w))
        rows.append(rec)
    return pd.DataFrame(rows)


# ---- 01: per-session traces, one figure per animal -------------------------
def plot_per_session_traces(long, taxis, low_n_sess):
    tcols = list(taxis)
    sessions = (long[["animal", "date", "sess", "group"]]
                .drop_duplicates().sort_values(["animal", "date"]))
    for animal in sorted(long["animal"].unique()):
        sa = sessions[sessions.animal == animal]
        n = len(sa)
        fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, sa.iterrows()):
            for cond in ["normal", "deviant"]:
                sub = long[(long.sess == row.sess) & (long.condition == cond)]
                if sub.empty:
                    continue
                mat = sub[tcols].to_numpy(float)
                mu = np.nanmean(mat, 0); se = sem(mat, 0)
                ax.plot(taxis, mu, color=COND_COLORS[cond], lw=1.6,
                        label=f"{cond} (n={len(sub)})")
                ax.fill_between(taxis, mu - se, mu + se,
                                color=COND_COLORS[cond], alpha=0.25, lw=0)
            flag = "  low-n" if row.sess in low_n_sess else ""
            ax.set_title(f"{row.date}\n[{row.group}]{flag}", fontsize=8,
                         color=("red" if row.sess in low_n_sess else "black"))
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("time from pattern onset (s)", fontsize=8)
            shade_pips(ax, row.group)
            ax.legend(fontsize=5.5, loc="upper left", frameon=False)
        axes[0].set_ylabel("z-scored pupil\n(baseline-corrected)")
        fig.suptitle(f"{animal} — normal vs deviant per session "
                     f"(red title = <{MIN_DEVIANTS} deviants; yellow band = "
                     "deviant tone(s): C, or B/C/D for control_newsound)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        f = os.path.join(OUTDIR, f"01_traces_per_session_{animal}.png")
        fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


# ---- 02b: plain grand-average by group (v1) --------------------------------
def plot_grandavg_by_group(sess_df, taxis, groups, tag):
    gps = [g for g in groups if g in set(sess_df["group"])]
    if not gps:
        return
    fig, axes = plt.subplots(1, len(gps), figsize=(4.6 * len(gps), 3.8), sharey=True)
    if len(gps) == 1:
        axes = [axes]
    for ax, grp in zip(axes, gps):
        for cond in ["normal", "deviant"]:
            sub = sess_df[(sess_df.group == grp) & (sess_df.condition == cond)]
            if sub.empty:
                continue
            tr = np.vstack(sub["mean_trace"].to_numpy())
            mu = np.nanmean(tr, 0); se = sem(tr, 0)
            ax.plot(taxis, mu, color=COND_COLORS[cond], lw=2,
                    label=f"{cond} ({len(sub)} sess)")
            ax.fill_between(taxis, mu - se, mu + se,
                            color=COND_COLORS[cond], alpha=0.25, lw=0)
        ax.set_title(grp, fontsize=11); ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("time from pattern onset (s)")
        shade_pips(ax, grp); ax.legend(fontsize=7, loc="upper left", frameon=False)
    axes[0].set_ylabel("z-scored pupil (session-mean ± SEM)")
    fig.suptitle(f"Grand-average pupil by group — unit = session (test phase) "
                 f"[{tag}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    f = os.path.join(OUTDIR, f"02b_grandavg_by_group_{tag}.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


# ---- 02: grand-average ALL vs low-n excluded (v2) ---------------------------
def plot_grandavg_sensitivity(sess_df, taxis, low_n_sess, groups, tag):
    gps = [g for g in groups if g in set(sess_df["group"])]
    if not gps:
        return
    fig, axes = plt.subplots(2, len(gps), figsize=(4.6 * len(gps), 7),
                             sharey=True, sharex=True)
    if len(gps) == 1:
        axes = axes.reshape(2, 1)
    for c, grp in enumerate(gps):
        for r, (label, keep_all) in enumerate(
                [("ALL sessions", True), (f">= {MIN_DEVIANTS} deviants", False)]):
            ax = axes[r, c]
            for cond in ["normal", "deviant"]:
                sub = sess_df[(sess_df.group == grp) & (sess_df.condition == cond)]
                if not keep_all:
                    sub = sub[~sub.sess.isin(low_n_sess)]
                if sub.empty:
                    continue
                tr = np.vstack(sub["mean_trace"].to_numpy())
                mu = np.nanmean(tr, 0); se = sem(tr, 0)
                ax.plot(taxis, mu, color=COND_COLORS[cond], lw=2,
                        label=f"{cond} ({len(sub)} sess)")
                ax.fill_between(taxis, mu - se, mu + se,
                                color=COND_COLORS[cond], alpha=0.25, lw=0)
            ax.axhline(0, color="k", lw=0.5); shade_pips(ax, grp)
            ax.legend(fontsize=6.5, loc="upper left", frameon=False)
            if r == 0:
                ax.set_title(grp, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{label}\nz-scored pupil")
            if r == 1:
                ax.set_xlabel("time from pattern onset (s)")
    fig.suptitle("Grand-average by group: ALL sessions (top) vs low-n excluded "
                 f"(bottom) — unit = session  [{tag}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    f = os.path.join(OUTDIR, f"02_grandavg_sensitivity_{tag}.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


def shade_pips_overlay(ax, groups):
    """For overlaid multi-group plots: shade the union of deviant pips across
    the groups shown. If groups disagree (e.g. C-only + B/C/D), the extra pips
    (B,D) are drawn with a lighter hatch-free tint and noted in the title."""
    dev_union = set().union(*[deviant_pips_for(g) for g in groups])
    for label, (a, b) in PIP_WINDOWS.items():
        p = label.split()[0]
        is_dev = p in dev_union
        ax.axvspan(a, b, color=("#F2C14E" if is_dev else "#000000"),
                   alpha=(0.18 if is_dev else 0.05), zorder=0)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    y0, y1 = ax.get_ylim()
    for label, (a, b) in PIP_WINDOWS.items():
        ax.text((a + b) / 2, y1 - 0.06 * (y1 - y0), label[0],
                ha="center", va="top", fontsize=8, fontweight="bold")


def _dev_note(groups):
    if any(g in DEVIANT_PIPS for g in groups):
        return "yellow = deviant tone(s): B/C/D for control_newsound, C otherwise"
    return "yellow = deviant C tone"


# ---- 03b: single-panel difference, cluster-shaded (v1) ---------------------
def plot_difference_traces_v1(sess_df, taxis, groups, tag):
    gps = [g for g in groups if g in set(sess_df["group"])]
    if not gps:
        return
    fig, ax = plt.subplots(figsize=(9, 4.4))
    sig_masks = {}
    for grp in gps:
        piv = {}
        for cond in ["normal", "deviant"]:
            s = sess_df[(sess_df.group == grp) & (sess_df.condition == cond)]
            piv[cond] = {r.sess: r.mean_trace for _, r in s.iterrows()}
        shared = sorted(set(piv["normal"]) & set(piv["deviant"]))
        if not shared:
            continue
        diff = np.vstack([piv["deviant"][s] - piv["normal"][s] for s in shared])
        mu = np.nanmean(diff, 0); se = sem(diff, 0)
        ax.plot(taxis, mu, color=GROUP_COLORS[grp], lw=2,
                label=f"{grp} ({len(shared)} sess)")
        ax.fill_between(taxis, mu - se, mu + se,
                        color=GROUP_COLORS[grp], alpha=0.18, lw=0)
        sig_masks[grp] = (cluster_perm_sig(diff, taxis) if len(shared) >= 3
                          else np.zeros(len(taxis), bool))
    ax.axhline(0, color="k", lw=0.6); shade_pips_overlay(ax, gps)
    y0, y1 = ax.get_ylim(); any_sig = False
    for grp, mask in sig_masks.items():
        if mask.any():
            any_sig = True
            ax.fill_between(taxis, y0, y1, where=mask,
                            color=SIGNIF_COLOR, alpha=SIGNIF_ALPHA, lw=0, zorder=0.2)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("time from pattern onset (s)")
    ax.set_ylabel("Δ pupil (deviant − normal), session-paired")
    sub = ("pink band = significant cluster (permutation p<.05); "
           if any_sig else "no significant cluster (permutation p<.05); ")
    ax.set_title("Deviant − normal difference by group (test phase)\n"
                 + sub + _dev_note(gps), fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    if any_sig:
        handles.append(Patch(facecolor=SIGNIF_COLOR, alpha=SIGNIF_ALPHA,
                             label="significant period"))
    ax.legend(handles=handles, fontsize=8, frameon=False)
    fig.tight_layout()
    f = os.path.join(OUTDIR, f"03b_difference_traces_by_group_{tag}.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


# ---- 03: two-panel difference ALL vs excluded (v2) -------------------------
def plot_difference_traces(sess_df, taxis, low_n_sess, groups, tag):
    gps = [g for g in groups if g in set(sess_df["group"])]
    if not gps:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    for ax, (label, keep_all) in zip(
            axes, [("ALL sessions", True), (f">= {MIN_DEVIANTS} deviants", False)]):
        sig_masks = {}
        for grp in gps:
            piv = {}
            for cond in ["normal", "deviant"]:
                s = sess_df[(sess_df.group == grp) & (sess_df.condition == cond)]
                if not keep_all:
                    s = s[~s.sess.isin(low_n_sess)]
                piv[cond] = {r.sess: r.mean_trace for _, r in s.iterrows()}
            shared = sorted(set(piv["normal"]) & set(piv["deviant"]))
            if not shared:
                continue
            diff = np.vstack([piv["deviant"][s] - piv["normal"][s] for s in shared])
            mu = np.nanmean(diff, 0); se = sem(diff, 0)
            ax.plot(taxis, mu, color=GROUP_COLORS[grp], lw=2,
                    label=f"{grp} ({len(shared)})")
            ax.fill_between(taxis, mu - se, mu + se,
                            color=GROUP_COLORS[grp], alpha=0.15, lw=0)
            sig_masks[grp] = (cluster_perm_sig(diff, taxis) if len(shared) >= 3
                              else np.zeros(len(taxis), bool))
        ax.axhline(0, color="k", lw=0.6); shade_pips_overlay(ax, gps)
        y0, y1 = ax.get_ylim()
        for grp, mask in sig_masks.items():
            if mask.any():
                ax.fill_between(taxis, y0, y1, where=mask,
                                color=SIGNIF_COLOR, alpha=SIGNIF_ALPHA, lw=0, zorder=0.2)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("time from pattern onset (s)")
        ax.set_title(label, fontsize=11)
        handles, labels_legend = ax.get_legend_handles_labels()
        if any(mask.any() for mask in sig_masks.values()):
            handles.append(Patch(facecolor=SIGNIF_COLOR, alpha=SIGNIF_ALPHA,
                                 label="significant period"))
        ax.legend(handles=handles, fontsize=7, frameon=False)
    axes[0].set_ylabel("Δ pupil (deviant − normal), session-paired")
    fig.suptitle(f"Deviant − normal by group  [{tag}]  "
                 "(pink band = significant cluster, permutation p<.05; "
                 + _dev_note(gps) + ")", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f = os.path.join(OUTDIR, f"03_difference_traces_sensitivity_{tag}.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


# ---- 04: scalar summaries --------------------------------------------------
def plot_scalar_summaries(sess_df, low_n_sess):
    metrics = [("max_resp", "max pupil in resp. window"),
               ("mean_resp", "mean pupil in resp. window")]
    gps = groups_present(sess_df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, (metric, ylabel) in zip(axes, metrics):
        positions, ticklabels, pos = [], [], 0
        for grp in gps:
            for cond in ["normal", "deviant"]:
                sub = sess_df[(sess_df.group == grp) & (sess_df.condition == cond)]
                vals = sub[metric].dropna().values
                is_low = sub.set_index("sess")[metric].index.isin(low_n_sess)
                pos += 1
                if len(vals):
                    bp = ax.boxplot(vals, positions=[pos], widths=0.6,
                                    patch_artist=True, showfliers=False)
                    for b in bp["boxes"]:
                        b.set(facecolor=COND_COLORS[cond], alpha=0.35)
                    for med in bp["medians"]:
                        med.set(color="k")
                    xj = pos + RNG.uniform(-0.12, 0.12, len(vals))
                    ax.scatter(xj, vals, s=20, color=COND_COLORS[cond],
                               edgecolor=np.where(is_low, "red", "k"),
                               linewidth=0.8, zorder=3)
                positions.append(pos); ticklabels.append(f"{grp}\n{cond}")
            pos += 0.6
        ax.set_xticks(positions); ax.set_xticklabels(ticklabels, fontsize=6)
        ax.set_ylabel(ylabel); ax.axhline(0, color="k", lw=0.5)
        ax.set_title(ylabel + f"\nwindow {RESPONSE_WINDOW[0]}–{RESPONSE_WINDOW[1]} s; "
                     "red edge = low-n session", fontsize=10)
    fig.suptitle("Per-session scalar pupil summaries (each dot = one session)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f = os.path.join(OUTDIR, "04_scalar_summaries.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


# ---- 05: per-session delta vs shuffle null ---------------------------------
def compute_delta_and_shuffle(long, taxis):
    recs = []
    tcols = list(taxis)
    for sess, sub in long.groupby("sess"):
        grp = sub["group"].iloc[0]; animal = sub["animal"].iloc[0]
        dev = sub[sub.condition == "deviant"][tcols].to_numpy(float)
        nor = sub[sub.condition == "normal"][tcols].to_numpy(float)
        if len(dev) < 3 or len(nor) < 3:
            continue
        obs = (np.nanmean(win_max(dev, taxis, RESPONSE_WINDOW))
               - np.nanmean(win_max(nor, taxis, RESPONSE_WINDOW)))
        pool = np.vstack([dev, nor]); nd = len(dev)
        null = np.empty(N_SHUFFLE); idx = np.arange(len(pool))
        for i in range(N_SHUFFLE):
            RNG.shuffle(idx)
            null[i] = (np.nanmean(win_max(pool[idx[:nd]], taxis, RESPONSE_WINDOW))
                       - np.nanmean(win_max(pool[idx[nd:]], taxis, RESPONSE_WINDOW)))
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_SHUFFLE + 1)
        recs.append({"sess": sess, "animal": animal, "group": grp,
                     "delta_max": obs, "null_lo": np.percentile(null, 2.5),
                     "null_hi": np.percentile(null, 97.5), "p_shuffle": p,
                     "n_dev": len(dev), "n_nor": len(nor),
                     "deviant_below_min": len(dev) < MIN_DEVIANTS})
    return pd.DataFrame(recs)


def plot_delta(delta_df):
    fig, ax = plt.subplots(figsize=(max(10, 0.42 * len(delta_df)), 4.8))
    pos = 0; xt, xl = [], []
    for grp in [g for g in GROUP_ORDER if g in set(delta_df["group"])]:
        sub = delta_df[delta_df.group == grp].sort_values("n_dev")
        for _, r in sub.iterrows():
            pos += 1
            ax.plot([pos, pos], [r.null_lo, r.null_hi], color="lightgray",
                    lw=6, solid_capstyle="round", zorder=1)
            ax.plot(pos, r.delta_max, "o", color=GROUP_COLORS[grp], ms=7, zorder=3,
                    markeredgecolor=("red" if r.deviant_below_min else "k"),
                    markeredgewidth=1.2)
            if r.p_shuffle < 0.05:
                ax.text(pos, r.delta_max, "*", ha="center", va="bottom", fontsize=12)
            xt.append(pos)
            xl.append(f"{r.animal}\n{r.sess.split('_')[1]}\n({r.n_dev})")
        pos += 0.8
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=5)
    ax.set_ylabel("Δ max pupil (deviant − normal)")
    ax.set_title("Per-session deviant−normal Δ vs 95% shuffle null (gray)\n"
                 "* p<.05 vs shuffle; red ring = low-n session; (n) = #deviants",
                 fontsize=11)
    for g in [g for g in GROUP_ORDER if g in set(delta_df["group"])]:
        ax.scatter([], [], color=GROUP_COLORS[g], label=g)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    f = os.path.join(OUTDIR, "05_delta_pupil_shuffle.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


# ============================================================================
# STATISTICS
# ============================================================================
def session_level_stats(sess_df, delta_df, exclude_low_n, low_n_sess, tag, lines):
    def W(s=""): lines.append(s)
    sdf, ddf = sess_df.copy(), delta_df.copy()
    if exclude_low_n:
        sdf = sdf[~sdf.sess.isin(low_n_sess)]
        ddf = ddf[~ddf.sess.isin(low_n_sess)]

    W("\n" + "#" * 78)
    W(f"SESSION-LEVEL STATISTICS  [{tag}]")
    W("#" * 78)

    W("\n[1] Deviant vs Normal within each group (paired per session, max pupil)")
    for grp in groups_present(sdf):
        piv = (sdf[sdf.group == grp]
               .pivot_table(index="sess", columns="condition",
                            values="max_resp").dropna())
        if len(piv) < 3:
            W(f"  {grp:18s}: n={len(piv)} — too few for a test"); continue
        d, n = piv["deviant"].values, piv["normal"].values
        try:
            _, wp = stats.wilcoxon(d, n)
        except ValueError:
            wp = np.nan
        _, tp = stats.ttest_rel(d, n)
        W(f"  {grp:18s}: n={len(piv):2d} sess | meanΔ={np.mean(d-n):+.3f} | "
          f"Wilcoxon p={wp:.4f} | paired t p={tp:.4f}")

    W("\n[2] Deviant effect (Δ) across groups (omnibus)")
    gps = [g for g in GROUP_ORDER if g in set(ddf["group"])]
    groups = [ddf[ddf.group == g]["delta_max"].dropna().values for g in gps]
    for g, arr in zip(gps, groups):
        sd = np.std(arr, ddof=1) if len(arr) > 1 else float("nan")
        mn = np.mean(arr) if len(arr) else float("nan")
        W(f"    {g:18s}: n={len(arr):2d}  meanΔ={mn:+.3f}  sd={sd:.3f}")
    valid = [a for a in groups if len(a) >= 2]
    if len(valid) >= 2:
        f_stat, f_p = stats.f_oneway(*valid)
        h_stat, h_p = stats.kruskal(*valid)
        W(f"    One-way ANOVA:  F={f_stat:.3f}, p={f_p:.4f}")
        W(f"    Kruskal-Wallis: H={h_stat:.3f}, p={h_p:.4f}")

    W("\n[3] Pairwise group comparisons of Δ (Mann-Whitney U)")
    for a, b in itertools.combinations(gps, 2):
        ga = ddf[ddf.group == a]["delta_max"].dropna().values
        gb = ddf[ddf.group == b]["delta_max"].dropna().values
        if len(ga) >= 2 and len(gb) >= 2:
            u, p = stats.mannwhitneyu(ga, gb, alternative="two-sided")
            W(f"    {a:18s} vs {b:18s}: U={u:6.1f}, p={p:.4f} (n={len(ga)},{len(gb)})")

    W("\n[4] Per-session shuffle test summary")
    for grp in gps:
        sub = ddf[ddf.group == grp]
        if len(sub):
            W(f"    {grp:18s}: {(sub.p_shuffle<0.05).sum()}/{len(sub)} "
              "sessions significant (p<.05)")

    W("\n[5] Pip-window profile (mean pupil per ABCD pip, session-averaged)")
    for grp in groups_present(sdf):
        W(f"    -- {grp} --")
        for cond in ["normal", "deviant"]:
            sub = sdf[(sdf.group == grp) & (sdf.condition == cond)]
            if sub.empty:
                continue
            vals = {p.split()[0]: sub[f"mean_{p.split()[0]}"].mean()
                    for p in PIP_WINDOWS}
            W("       %-8s " % cond + "  ".join(f"{k}={v:+.3f}"
                                                for k, v in vals.items()))


def build_trial_level(long, taxis):
    tcols = list(taxis)
    mat = long[tcols].to_numpy(float)
    m = (taxis > RESPONSE_WINDOW[0]) & (taxis <= RESPONSE_WINDOW[1])
    with np.errstate(all="ignore"):
        mx = np.nanmax(mat[:, m], axis=1)
        mn = np.nanmean(mat[:, m], axis=1)
    out = long[["animal", "date", "sess", "group", "condition"]].copy()
    out["max_resp"] = mx; out["mean_resp"] = mn
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["max_resp"])


def run_mixed_model(trial_df, exclude_low_n, low_n_sess, tag, lines):
    def W(s=""): lines.append(s)
    df = trial_df.copy()
    if exclude_low_n:
        df = df[~df.sess.isin(low_n_sess)]
    gps = [g for g in GROUP_ORDER if g in set(df["group"])]
    df["condition"] = pd.Categorical(df["condition"],
                                     categories=["normal", "deviant"])
    df["group"] = pd.Categorical(df["group"], categories=gps)
    W(f"\n--- Mixed model [{tag}]: n_trials={len(df)}, "
      f"n_sessions={df.sess.nunique()}, n_animals={df.animal.nunique()} ---")
    W("    max_resp ~ condition * group, groups=animal, session variance comp.")
    try:
        mf = smf.mixedlm("max_resp ~ condition * group", df, groups=df["animal"],
                         vc_formula={"sess": "0 + C(sess)"},
                         re_formula="1").fit(method="lbfgs", maxiter=300, disp=False)
        for k in mf.params.index:
            if ("condition" in k or "Intercept" in k) and "Var" not in k:
                W(f"      {k:52s} beta={mf.params[k]:+.4f}  p={mf.pvalues[k]:.4f}")
        W(f"      (converged={mf.converged})")
    except Exception as e:
        W(f"      mixedlm failed ({e}); OLS fallback:")
        mf = smf.ols("max_resp ~ condition * group", df).fit()
        for k in mf.params.index:
            if "condition" in k or "Intercept" in k:
                W(f"      {k:52s} beta={mf.params[k]:+.4f}  p={mf.pvalues[k]:.4f}")


# ============================================================================
def main():
    lines = []
    def W(s=""): lines.append(s); print(s)

    long = load_long()
    taxis = time_axis(long)

    counts = build_count_table(long)
    counts.to_csv(os.path.join(OUTDIR, "per_session_counts.csv"), index=False)
    low_n_sess = set(counts.loc[counts.deviant_below_min, "sess"])
    plot_count_table(counts)

    W("=" * 78)
    W("PUPIL DEVIANT ANALYSIS — expanded dataset, TEST PHASE ONLY")
    W("=" * 78)
    W(f"Animals: {sorted(long['animal'].unique())}")
    W(f"Sessions: {long['sess'].nunique()} | trials: {len(long)}")
    W(f"Low-n threshold: {MIN_DEVIANTS} deviants | "
      f"response window {RESPONSE_WINDOW[0]}–{RESPONSE_WINDOW[1]} s")
    W(f"Low-n sessions (n={len(low_n_sess)}): "
      + (", ".join(sorted(low_n_sess)) if low_n_sess else "none"))
    W("\nGroup composition:")
    for grp in groups_present(counts):
        sub = counts[counts.group == grp]
        W(f"  {grp:18s}: {len(sub):2d} sessions | animals "
          f"{sorted(sub.animal.unique())} | dates {sorted(sub.date.unique())}")
        W(f"  {'':18s}  deviant n {sub.n_deviant.min()}–{sub.n_deviant.max()} "
          f"(median {int(sub.n_deviant.median())}), "
          f"low-n={int(sub.deviant_below_min.sum())}")

    plot_per_session_traces(long, taxis, low_n_sess)
    sess_df = session_means(long, taxis)
    sess_df.drop(columns=["mean_trace"]).to_csv(
        os.path.join(OUTDIR, "per_session_summaries.csv"), index=False)

    # Two paired figures per plot type, per user request:
    #   file 1 = control + opto ; file 2 = opto_late + control_newsound
    GROUP_PAIRS = [(["control", "opto"], "control_opto"),
                   (["opto_late", "control_newsound"], "optolate_controlnewsound")]
    for groups, tag in GROUP_PAIRS:
        plot_grandavg_by_group(sess_df, taxis, groups, tag)
        plot_grandavg_sensitivity(sess_df, taxis, low_n_sess, groups, tag)
        plot_difference_traces_v1(sess_df, taxis, groups, tag)
        plot_difference_traces(sess_df, taxis, low_n_sess, groups, tag)
    plot_scalar_summaries(sess_df, low_n_sess)

    delta_df = compute_delta_and_shuffle(long, taxis)
    delta_df.to_csv(os.path.join(OUTDIR, "per_session_delta_shuffle.csv"),
                    index=False)
    plot_delta(delta_df)

    session_level_stats(sess_df, delta_df, False, low_n_sess,
                        "ALL sessions", lines)
    session_level_stats(sess_df, delta_df, True, low_n_sess,
                        f"EXCLUDING <{MIN_DEVIANTS} deviants", lines)

    trial_df = build_trial_level(long, taxis)
    trial_df.to_csv(os.path.join(OUTDIR, "trial_level_max_resp.csv"), index=False)
    lines.append("\n" + "#" * 78)
    lines.append("TRIAL-LEVEL LINEAR MIXED-EFFECTS MODELS")
    lines.append("#" * 78)
    run_mixed_model(trial_df, False, low_n_sess, "ALL sessions", lines)
    run_mixed_model(trial_df, True, low_n_sess,
                    f"EXCLUDING <{MIN_DEVIANTS} deviants", lines)

    lines.append("\nNotes:")
    lines.append(" - NA04/NA05 are all opto (they only run 260720/260721).")
    lines.append(" - control merges 260606/607 (original sounds) with "
                 "260717/720/721 (new sounds).")
    lines.append("   NA01-03 only and use a DIFFERENT sound set, so they are kept")
    lines.append("   as separate groups and never merged with the original control.")
    lines.append(" - Deviant tone (C) = 0.50-0.75 s; divergence expected from there.")
    lines.append(" - Compare ALL vs EXCLUDED blocks to check low-n sessions are not")
    lines.append("   driving any result.")

    with open(os.path.join(OUTDIR, "statistics_summary.txt"), "w") as fh:
        fh.write("\n".join(lines))
    print("\nDONE. Outputs in", OUTDIR)


if __name__ == "__main__":
    main()