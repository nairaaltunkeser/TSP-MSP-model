#!/usr/bin/env python3
# ============================================================================
#  BEHAVIOURAL CONTROL ANALYSIS — does the OPTO manipulation change behaviour?
#
#  PURPOSE
#  -------
#  The pupil analysis compares deviant vs normal responses across control /
#  opto / opto_late / control_newsound sessions. Before interpreting any pupil
#  difference as a perceptual or attentional effect, we need to rule out the
#  trivial explanation that opto simply changed TASK PERFORMANCE (the animal
#  licked less, licked later, or missed more trials). This script tests exactly
#  that, using the behavioural trial table (EC_opto_td_df2.h5).
#
#  If behaviour is statistically indistinguishable between opto and control,
#  the pupil differences cannot be attributed to a gross behavioural change.
#
#  MEASURES (all computed per trial, then summarised per session)
#  -------------------------------------------------------------
#    1. hit_rate        - fraction of trials with Trial_Outcome == 1.
#                         Verified coding: outcome 1 trials ALWAYS have a
#                         RewardTone_Time, outcome 0 trials NEVER do, so
#                         1 = HIT (rewarded), 0 = MISS.
#    2. first_lick_lat  - latency (s) from ToneTime to the first lick at or
#                         after tone onset. NaN if the animal never licked.
#    3. lick_lat_reward - latency (s) from ToneTime to the reward tone
#                         (hit trials only) -- a second, independent timing read.
#    4. n_licks         - total licks recorded in the trial.
#    5. n_licks_post    - licks occurring at/after tone onset.
#    6. early_licks     - the Early_Licks counter (premature licking).
#    7. p_lick          - fraction of trials with >= 1 post-tone lick.
#
#  Lick_Times is a ';'-separated list of 'HH:MM:SS.ffffff' clock stamps; all
#  times are converted to seconds-since-midnight and referenced to ToneTime of
#  the same trial, so the clock origin cancels out.
#
#  SCOPE
#  -----
#  Pattern trials only (N_TonesPlayed == 4), TEST PHASE only (trial_num >= 101)
#  -- i.e. exactly the trials that enter the pupil analysis.
#
#  GROUPING — identical to pupil_analysis_main.py:
#    * NA04 / NA05                                  -> 'opto'
#    * NA01-03: 260606/260607/260717/260720/260721  -> 'control'  (merged)
#               260710/260713/260715                -> 'control_newsound'
#               260618/260619/260620                -> 'opto_late'
#               otherwise                           -> 'opto'
#      NA03 exception: 260606/260607 are 'opto' for NA03 only.
#  NOTE: the behavioural table's 'sess' has no run suffix, so sessions are
#  keyed by animal+date (unambiguous here).
#
#  STATISTICS
#  ----------
#    - Per-group descriptives (session-level mean +/- SEM).
#    - Omnibus across groups: one-way ANOVA + Kruskal-Wallis on session means.
#    - Pairwise Mann-Whitney U, with the OPTO-vs-CONTROL contrast highlighted
#      as the key comparison.
#    - Deviant vs normal within group (paired per session) for hit rate and
#      latency -- checks the deviant itself does not disrupt behaviour.
#    - Trial-level mixed models that respect the nesting:
#         logistic  hit ~ group (+ animal random intercept)   [correctness]
#         linear    first_lick_lat ~ group (+ animal random)  [timing]
#    - Equivalence framing: because this is a control analysis, a NON-significant
#      result is the desired outcome, so 95% CIs on the opto-minus-control
#      difference are reported to show the effect is small, not merely untested.
#
#  OUTPUTS -> outputs_behaviour/
#    B1_hitrate_by_group.png        hit rate per session, by group & condition
#    B2_lick_latency_by_group.png   first-lick latency distributions
#    B3_lick_counts_by_group.png    lick counts / early licks
#    B4_latency_histograms.png      pooled latency distributions per group
#    B5_session_timecourse.png      hit rate across trial number within session
#    behaviour_statistics.txt       all tests
#    per_session_behaviour.csv      session-level summary table
#    per_trial_behaviour.csv        trial-level table (for re-analysis)
# ============================================================================

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

# ----------------------------------------------------------------------------
TD_H5    = "EC_opto_td_df.h5"
PUPIL_H5 = "EC_opto_pupil.h5"     # only used to match the pupil session set
OUTDIR   = "outputs_behaviour_control_opto"

TEST_PHASE_MIN_TRIAL = 101
PATTERN_N_TONES      = 4

# The behavioural table contains more sessions than the pupil file (e.g. date
# 260722, and sessions with no usable pupil recording). For a control analysis
# that speaks directly to the pupil results, restrict to the SAME sessions the
# pupil analysis uses. Set False to analyse all behavioural sessions instead.
MATCH_PUPIL_SESSIONS = True

NA45_ANIMALS       = {"NA04", "NA05"}
NA45_OPTO_DATES    = {"260720", "260721", "260722"}   # light-on  = opto
NA45_CONTROL_DATES = {"260723", "260724", "260725"}   # lights off = control
CONTROL_DATES          = {"260606", "260607", "260717", "260720", "260721", "260722"}
NA03_OPTO_DATES        = {"260606", "260607"}
OPTOLATE_DATES         = {"260618", "260619", "260620"}
CONTROL_NEWSOUND_DATES = {"260710", "260713", "260715"}
GROUP_ORDER = ["control", "opto"]   # <-- restricted to control + opto only

GROUP_COLORS = {"control": "#7F7F7F", "opto": "#2CA02C",
                "opto_late": "#9467BD", "control_newsound": "#D62728"}
COND_COLORS = {"normal": "#4C72B0", "deviant": "#C44E52"}
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# Robust file location
# ----------------------------------------------------------------------------
# Resolve data files relative to THIS script's folder, so the script works no
# matter what the current working directory is (IDEs often differ from the
# script location). If a file is missing we print the folder contents and try a
# case-insensitive match, which catches the usual culprits: hidden ".h5.h5"
# double extensions, stray spaces, or capitalisation differences.
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
    # case-insensitive / whitespace-tolerant match
    target = fname.lower().replace(" ", "")
    for e in entries:
        if e.lower().replace(" ", "") == target:
            print(f"NOTE: using '{e}' for requested '{fname}'.")
            return os.path.join(_HERE, e)
    # helpful diagnostic
    h5 = [e for e in entries if e.lower().endswith((".h5", ".hdf5"))]
    msg = [f"Could not find '{fname}' in:", f"  {_HERE}", ""]
    if h5:
        msg.append("HDF5-like files that ARE present:")
        msg += [f"  {repr(e)}" for e in h5]
        msg.append("")
        msg.append("Note the exact spelling above (repr shows hidden spaces).")
    else:
        msg.append("No .h5/.hdf5 files found in this folder at all.")
        msg.append("Folder contents:")
        msg += [f"  {repr(e)}" for e in entries[:40]]
    raise FileNotFoundError("\n".join(msg))


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


def clock_to_sec(t):
    """'HH:MM:SS.ffffff' -> seconds since midnight. Returns NaN if unparseable
    or if the field is the '00:00:00' placeholder used for 'not recorded'."""
    s = str(t)
    if s in ("00:00:00", "nan", "None", ""):
        return np.nan
    try:
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except Exception:
        return np.nan


def parse_licks(v):
    if not isinstance(v, str) or len(v) < 4:
        return []
    out = []
    for x in v.split(";"):
        if x:
            t = clock_to_sec(x)
            if not np.isnan(t):
                out.append(t)
    return out


def load_trials():
    with pd.HDFStore(resolve_data_file(TD_H5), "r") as store:
        df = store["/df"]
    idx = df.index.to_frame(index=False)
    d = df.reset_index(drop=True)
    for c in ["name", "date", "trial_num"]:
        d[c] = idx[c].values
    d = d.rename(columns={"name": "animal"})
    d["date"] = d["date"].astype(str)

    # scope: pattern trials, test phase
    n0 = len(d)
    d = d[(d["N_TonesPlayed"] == PATTERN_N_TONES) &
          (d["trial_num"] >= TEST_PHASE_MIN_TRIAL)].copy()
    print(f"Scope filter (4-tone pattern trials, trial >= "
          f"{TEST_PHASE_MIN_TRIAL}): {n0} -> {len(d)} trials.")

    d["group"] = [classify_session(a, dt)
                  for a, dt in zip(d["animal"], d["date"])]
    d["sess_key"] = d["animal"] + "_" + d["date"]

    # ---- restrict to control + opto only (drop opto_late / control_newsound) ----
    before_g = d["sess_key"].nunique()
    d = d[d["group"].isin(GROUP_ORDER)].copy()
    print(f"Restricted to {GROUP_ORDER}: {before_g} -> "
          f"{d['sess_key'].nunique()} sessions.")

    # restrict to the sessions that also have pupil data, so this control
    # analysis speaks to exactly the sessions in the pupil results
    if MATCH_PUPIL_SESSIONS:
        with pd.HDFStore(resolve_data_file(PUPIL_H5), "r") as ps:
            psess = set(ps["/normal"].index.to_frame(index=False)["sess"])
        # pupil sess is animal_date_run -> reduce to animal_date
        pkeys = {"_".join(s.split("_")[:2]) for s in psess}
        before = d["sess_key"].nunique()
        d = d[d["sess_key"].isin(pkeys)].copy()
        print(f"Matched to pupil sessions: {before} -> "
              f"{d['sess_key'].nunique()} sessions.")
    d["condition"] = np.where(d["Pattern_Type"] == 1, "deviant", "normal")

    # ---- behavioural measures ----
    d["hit"] = (d["Trial_Outcome"] == 1).astype(int)
    d["tone_s"] = d["ToneTime"].map(clock_to_sec)
    d["reward_s"] = d["RewardTone_Time"].map(clock_to_sec)
    licks = d["Lick_Times"].map(parse_licks)
    d["n_licks"] = licks.map(len)

    def post_licks(lk, t0):
        if np.isnan(t0):
            return []
        return [t - t0 for t in lk if t >= t0]

    post = [post_licks(lk, t0) for lk, t0 in zip(licks, d["tone_s"])]
    d["n_licks_post"] = [len(p) for p in post]
    d["first_lick_lat"] = [p[0] if p else np.nan for p in post]
    d["p_lick"] = (d["n_licks_post"] > 0).astype(int)
    d["lick_lat_reward"] = d["reward_s"] - d["tone_s"]
    d.loc[d["lick_lat_reward"] < 0, "lick_lat_reward"] = np.nan
    d["early_licks"] = d["Early_Licks"]

    keep = ["animal", "date", "sess_key", "group", "condition", "trial_num",
            "hit", "first_lick_lat", "lick_lat_reward", "n_licks",
            "n_licks_post", "p_lick", "early_licks"]
    return d[keep].reset_index(drop=True)


MEASURES = [
    ("hit",             "hit rate (P correct)"),
    ("first_lick_lat",  "first-lick latency (s from tone)"),
    ("lick_lat_reward", "reward-tone latency (s from tone)"),
    ("n_licks",         "licks per trial"),
    ("n_licks_post",    "post-tone licks per trial"),
    ("p_lick",          "P(any post-tone lick)"),
    ("early_licks",     "early (premature) licks"),
]


def session_summary(trials):
    """Collapse to one row per session (and per session x condition)."""
    aggs = {m: "mean" for m, _ in MEASURES}
    by_sess = (trials.groupby(["animal", "date", "sess_key", "group"])
               .agg(**{m: (m, "mean") for m, _ in MEASURES},
                    n_trials=("hit", "size")).reset_index())
    by_sc = (trials.groupby(["animal", "date", "sess_key", "group", "condition"])
             .agg(**{m: (m, "mean") for m, _ in MEASURES},
                  n_trials=("hit", "size")).reset_index())
    return by_sess, by_sc


def groups_present(df):
    return [g for g in GROUP_ORDER if g in set(df["group"])]


def sem(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan


# ----------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------
def _box_by_group(ax, by_sess, metric, ylabel):
    gps = groups_present(by_sess)
    data = [by_sess[by_sess.group == g][metric].dropna().values for g in gps]
    bp = ax.boxplot(data, positions=range(len(gps)), widths=0.6,
                    patch_artist=True, showfliers=False)
    for b, g in zip(bp["boxes"], gps):
        b.set(facecolor=GROUP_COLORS[g], alpha=0.35)
    for med in bp["medians"]:
        med.set(color="k")
    for i, (g, arr) in enumerate(zip(gps, data)):
        xj = i + RNG.uniform(-0.12, 0.12, len(arr))
        ax.scatter(xj, arr, s=22, color=GROUP_COLORS[g],
                   edgecolor="k", linewidth=0.4, zorder=3)
    ax.set_xticks(range(len(gps)))
    ax.set_xticklabels(gps, fontsize=7, rotation=15)
    ax.set_ylabel(ylabel, fontsize=9)


def plot_hitrate(by_sess, by_sc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    _box_by_group(axes[0], by_sess, "hit", "hit rate (P correct)")
    axes[0].set_title("Hit rate per session, by group\n(each dot = one session)",
                      fontsize=10)
    axes[0].set_ylim(0, 1.02); axes[0].axhline(0.5, color="k", lw=0.5, ls=":")

    ax = axes[1]
    gps = groups_present(by_sc)
    pos = 0; xt, xl = [], []
    for g in gps:
        for cond in ["normal", "deviant"]:
            vals = by_sc[(by_sc.group == g) &
                         (by_sc.condition == cond)]["hit"].dropna().values
            pos += 1
            if len(vals):
                bp = ax.boxplot(vals, positions=[pos], widths=0.6,
                                patch_artist=True, showfliers=False)
                for b in bp["boxes"]:
                    b.set(facecolor=COND_COLORS[cond], alpha=0.35)
                for med in bp["medians"]:
                    med.set(color="k")
                ax.scatter(pos + RNG.uniform(-0.12, 0.12, len(vals)), vals,
                           s=18, color=COND_COLORS[cond], edgecolor="k",
                           linewidth=0.3, zorder=3)
            xt.append(pos); xl.append(f"{g}\n{cond}")
        pos += 0.6
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6)
    ax.set_ylabel("hit rate"); ax.set_ylim(0, 1.02)
    ax.set_title("Hit rate by group x trial type\n"
                 "(deviant should not disrupt performance)", fontsize=10)
    fig.suptitle("BEHAVIOURAL CONTROL — task correctness", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f = os.path.join(OUTDIR, "B1_hitrate_by_group.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


def plot_latency(by_sess, by_sc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    _box_by_group(axes[0], by_sess, "first_lick_lat",
                  "first-lick latency (s)")
    axes[0].set_title("First-lick latency per session\n(from tone onset)",
                      fontsize=10)
    _box_by_group(axes[1], by_sess, "lick_lat_reward",
                  "reward-tone latency (s)")
    axes[1].set_title("Reward-tone latency per session\n(hit trials)", fontsize=10)
    fig.suptitle("BEHAVIOURAL CONTROL — response timing", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f = os.path.join(OUTDIR, "B2_lick_latency_by_group.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


def plot_lick_counts(by_sess):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (m, lab) in zip(axes, [("n_licks", "licks per trial"),
                                   ("n_licks_post", "post-tone licks"),
                                   ("early_licks", "early (premature) licks")]):
        _box_by_group(ax, by_sess, m, lab)
        ax.set_title(lab, fontsize=10)
    fig.suptitle("BEHAVIOURAL CONTROL — licking vigour", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f = os.path.join(OUTDIR, "B3_lick_counts_by_group.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


def plot_latency_hist(trials):
    gps = groups_present(trials)
    fig, axes = plt.subplots(1, len(gps), figsize=(3.6 * len(gps), 3.6),
                             sharex=True, sharey=True)
    if len(gps) == 1:
        axes = [axes]
    bins = np.linspace(0, 10, 41)
    for ax, g in zip(axes, gps):
        v = trials[trials.group == g]["first_lick_lat"].dropna().values
        ax.hist(v, bins=bins, color=GROUP_COLORS[g], alpha=0.75,
                density=True, edgecolor="white", linewidth=0.3)
        if len(v):
            ax.axvline(np.median(v), color="k", ls="--", lw=1.2,
                       label=f"median {np.median(v):.2f}s")
            ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"{g}\n(n={len(v)} trials)", fontsize=10)
        ax.set_xlabel("first-lick latency (s)")
    axes[0].set_ylabel("density")
    fig.suptitle("Pooled first-lick latency distributions "
                 "(shape differences would indicate a behavioural effect)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    f = os.path.join(OUTDIR, "B4_latency_histograms.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


def plot_session_timecourse(trials):
    """Hit rate as a function of position within the test phase -- checks that
    opto does not cause a progressive decline (e.g. fatigue/disengagement)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for g in groups_present(trials):
        sub = trials[trials.group == g].copy()
        # bin trials into deciles of within-session position
        sub["rank"] = sub.groupby("sess_key")["trial_num"].rank(pct=True)
        bins = np.linspace(0, 1, 11)
        sub["bin"] = pd.cut(sub["rank"], bins, include_lowest=True)
        # session-level means first, then average across sessions
        per_sess = sub.groupby(["sess_key", "bin"], observed=True)["hit"].mean()
        m = per_sess.groupby("bin", observed=True).mean()
        e = per_sess.groupby("bin", observed=True).apply(
            lambda x: sem(x.values))
        centers = [(b.left + b.right) / 2 for b in m.index]
        ax.errorbar(centers, m.values, yerr=e.values, color=GROUP_COLORS[g],
                    marker="o", ms=4, lw=1.6, capsize=2, label=g)
    ax.set_xlabel("relative position within test phase (0 = start, 1 = end)")
    ax.set_ylabel("hit rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("Hit rate across the session, by group\n"
                 "(a downward opto slope would indicate progressive "
                 "disengagement)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    f = os.path.join(OUTDIR, "B5_session_timecourse.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


# ----------------------------------------------------------------------------
# STATISTICS
# ----------------------------------------------------------------------------
def diff_ci(a, b, n_boot=5000, seed=0):
    """Bootstrap 95% CI on mean(a) - mean(b) (a = opto, b = control)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    d = np.mean(a) - np.mean(b)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = (np.mean(rng.choice(a, len(a), replace=True))
                   - np.mean(rng.choice(b, len(b), replace=True)))
    return d, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def run_stats(trials, by_sess, by_sc, lines):
    def W(s=""): lines.append(s); print(s)

    W("=" * 78)
    W("BEHAVIOURAL CONTROL ANALYSIS — does opto change task performance?")
    W("=" * 78)
    W("Scope: 4-tone pattern trials, test phase (trial >= "
      f"{TEST_PHASE_MIN_TRIAL}).")
    W(f"Trials: {len(trials)} | sessions: {by_sess.sess_key.nunique()} | "
      f"animals: {trials.animal.nunique()}")
    W("Trial_Outcome coding verified: 1 = HIT (reward tone present), 0 = MISS.")
    W("NOTE: a NON-significant result here is the DESIRED outcome -- it means")
    W("      behaviour cannot explain any pupil difference between groups.")

    W("\nGroup composition (sessions):")
    for g in groups_present(by_sess):
        sub = by_sess[by_sess.group == g]
        W(f"  {g:18s}: {len(sub):2d} sessions | animals "
          f"{sorted(sub.animal.unique())}")

    # ---- 1. descriptives + omnibus per measure ----
    W("\n" + "#" * 78)
    W("[1] SESSION-LEVEL DESCRIPTIVES AND OMNIBUS TESTS")
    W("#" * 78)
    for m, lab in MEASURES:
        W(f"\n-- {lab} --")
        gps = groups_present(by_sess)
        arrs = []
        for g in gps:
            v = by_sess[by_sess.group == g][m].dropna().values
            arrs.append(v)
            W(f"    {g:18s}: n={len(v):2d}  mean={np.mean(v):7.3f}  "
              f"SEM={sem(v):.3f}  median={np.median(v):7.3f}")
        valid = [a for a in arrs if len(a) >= 2]
        if len(valid) >= 2:
            f_stat, f_p = stats.f_oneway(*valid)
            h_stat, h_p = stats.kruskal(*valid)
            flag = "" if min(f_p, h_p) >= 0.05 else "   <-- SIGNIFICANT"
            W(f"    ANOVA F={f_stat:.3f} p={f_p:.4f} | "
              f"Kruskal H={h_stat:.3f} p={h_p:.4f}{flag}")

    # ---- 2. key contrast: opto vs control ----
    W("\n" + "#" * 78)
    W("[2] KEY CONTRAST — OPTO vs CONTROL (session level)")
    W("    Mann-Whitney U + bootstrap 95% CI on the difference in means.")
    W("    A CI spanning 0 supports 'no behavioural difference'.")
    W("#" * 78)
    for m, lab in MEASURES:
        a = by_sess[by_sess.group == "opto"][m].dropna().values
        b = by_sess[by_sess.group == "control"][m].dropna().values
        if len(a) >= 2 and len(b) >= 2:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            d, lo, hi = diff_ci(a, b)
            flag = "" if p >= 0.05 else "   <-- SIGNIFICANT"
            W(f"  {lab:34s}: diff(opto-control)={d:+.3f} "
              f"95%CI [{lo:+.3f}, {hi:+.3f}]  U={u:.1f} p={p:.4f}{flag}")

    # ---- 3. all pairwise ----
    W("\n" + "#" * 78)
    W("[3] ALL PAIRWISE GROUP COMPARISONS (Mann-Whitney U)")
    W("#" * 78)
    for m, lab in MEASURES:
        W(f"\n-- {lab} --")
        gps = groups_present(by_sess)
        for g1, g2 in itertools.combinations(gps, 2):
            a = by_sess[by_sess.group == g1][m].dropna().values
            b = by_sess[by_sess.group == g2][m].dropna().values
            if len(a) >= 2 and len(b) >= 2:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                flag = "" if p >= 0.05 else "  <-- SIG"
                W(f"    {g1:18s} vs {g2:18s}: U={u:6.1f} p={p:.4f}{flag}")

    # ---- 4. deviant vs normal within group (paired) ----
    W("\n" + "#" * 78)
    W("[4] DEVIANT vs NORMAL WITHIN GROUP (paired per session)")
    W("    Checks the deviant itself does not disrupt behaviour.")
    W("#" * 78)
    for m, lab in [("hit", "hit rate"),
                   ("first_lick_lat", "first-lick latency"),
                   ("n_licks_post", "post-tone licks")]:
        W(f"\n-- {lab} --")
        for g in groups_present(by_sc):
            piv = (by_sc[by_sc.group == g]
                   .pivot_table(index="sess_key", columns="condition",
                                values=m).dropna())
            if len(piv) < 3:
                W(f"    {g:18s}: n={len(piv)} — too few"); continue
            try:
                _, wp = stats.wilcoxon(piv["deviant"], piv["normal"])
            except ValueError:
                wp = np.nan
            _, tp = stats.ttest_rel(piv["deviant"], piv["normal"])
            d = (piv["deviant"] - piv["normal"]).mean()
            flag = "" if (np.isnan(wp) or wp >= 0.05) else "  <-- SIG"
            W(f"    {g:18s}: n={len(piv):2d} meanΔ(dev-nor)={d:+.3f} "
              f"Wilcoxon p={wp:.4f} paired t p={tp:.4f}{flag}")

    # ---- 5. trial-level mixed models ----
    W("\n" + "#" * 78)
    W("[5] TRIAL-LEVEL MIXED MODELS (animal as random effect)")
    W("    Uses every trial and respects trials-in-sessions-in-animals nesting.")
    W("#" * 78)
    df = trials.copy()
    gps = groups_present(df)
    df["group"] = pd.Categorical(df["group"], categories=gps)

    W("\n-- correctness: logistic  hit ~ group  (+ animal random intercept) --")
    try:
        mdl = smf.mixedlm("hit ~ group", df, groups=df["animal"],
                          vc_formula={"sess": "0 + C(sess_key)"},
                          re_formula="1").fit(method="lbfgs", maxiter=300,
                                              disp=False)
        for k in mdl.params.index:
            if "group" in k or "Intercept" in k:
                if "Var" not in k:
                    flag = "" if mdl.pvalues[k] >= 0.05 else "  <-- SIG"
                    W(f"    {k:34s} beta={mdl.params[k]:+.4f} "
                      f"p={mdl.pvalues[k]:.4f}{flag}")
        W(f"    (LMM on binary outcome = linear probability model; "
          f"converged={mdl.converged})")
    except Exception as e:
        W(f"    mixed model failed ({e})")

    W("\n-- timing: first_lick_lat ~ group (+ animal random intercept) --")
    d2 = df.dropna(subset=["first_lick_lat"])
    try:
        mdl = smf.mixedlm("first_lick_lat ~ group", d2, groups=d2["animal"],
                          vc_formula={"sess": "0 + C(sess_key)"},
                          re_formula="1").fit(method="lbfgs", maxiter=300,
                                              disp=False)
        for k in mdl.params.index:
            if "group" in k or "Intercept" in k:
                if "Var" not in k:
                    flag = "" if mdl.pvalues[k] >= 0.05 else "  <-- SIG"
                    W(f"    {k:34s} beta={mdl.params[k]:+.4f} "
                      f"p={mdl.pvalues[k]:.4f}{flag}")
        W(f"    (converged={mdl.converged})")
    except Exception as e:
        W(f"    mixed model failed ({e})")

    W("\n" + "#" * 78)
    W("INTERPRETATION GUIDE")
    W("#" * 78)
    W(" - If [2] shows no significant opto-vs-control difference AND the 95% CIs")
    W("   are narrow around 0, behaviour is matched and cannot explain pupil")
    W("   differences between those groups.")
    W(" - If [4] shows no deviant-vs-normal behavioural difference, the deviant")
    W("   pupil response is not confounded by a change in licking/correctness.")
    W(" - Any measure flagged SIGNIFICANT should be reported as a caveat and,")
    W("   ideally, added as a covariate to the pupil model.")

    return lines


# ----------------------------------------------------------------------------
def main():
    lines = []
    trials = load_trials()
    by_sess, by_sc = session_summary(trials)

    trials.to_csv(os.path.join(OUTDIR, "per_trial_behaviour.csv"), index=False)
    by_sess.to_csv(os.path.join(OUTDIR, "per_session_behaviour.csv"), index=False)
    by_sc.to_csv(os.path.join(OUTDIR, "per_session_condition_behaviour.csv"),
                 index=False)

    plot_hitrate(by_sess, by_sc)
    plot_latency(by_sess, by_sc)
    plot_lick_counts(by_sess)
    plot_latency_hist(trials)
    plot_session_timecourse(trials)

    run_stats(trials, by_sess, by_sc, lines)
    with open(os.path.join(OUTDIR, "behaviour_statistics.txt"), "w") as fh:
        fh.write("\n".join(lines))
    print("\nDONE. Outputs in", OUTDIR)


if __name__ == "__main__":
    main()
