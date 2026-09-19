#!/usr/bin/env python3
# Early vs late pupil responses during the test phase.
# Control and opto sessions only; compares normal and deviant trials.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

PUPIL_H5 = "EC_opto_pupil.h5"
OUTDIR   = "outputs_adaptation_control_opto"


SPLIT_MODE   = "fixed"      # "fixed" (thesis, first/last N) or "median"
N_EARLY_LATE = 7            # thesis N
TEST_PHASE_MIN_TRIAL = 101
MIN_PER_HALF = 4            # for median mode: min trials per half to test

NA45_ANIMALS      = {"NA04", "NA05"}
NA45_OPTO_DATES   = {"260720", "260721", "260722"}   # light-on  = opto
NA45_CONTROL_DATES= {"260723", "260724", "260725"}   # lights off = control
CONTROL_DATES          = {"260606", "260607", "260717", "260720", "260721", "260722"}
NA03_OPTO_DATES        = {"260606", "260607"}   # NA03 only: these are opto
OPTOLATE_DATES         = {"260618", "260619", "260620"}
CONTROL_NEWSOUND_DATES = {"260710", "260713", "260715"}
GROUP_ORDER = ["control", "opto"]   # <-- restricted to control + opto only

PIP_WINDOWS = {"A (0.00-0.25)": (0.00, 0.25), "B (0.25-0.50)": (0.25, 0.50),
               "C (0.50-0.75)": (0.50, 0.75), "D (0.75-1.00)": (0.75, 1.00)}
RESPONSE_WINDOW = (0.50, 2.00)

HALF_COLORS = {"early": "#E1812C", "late": "#3A76AF"}
GROUP_COLORS = {"control": "#7F7F7F", "opto": "#2CA02C", "opto_late": "#9467BD", "control_newsound": "#D62728"}
COND_TITLE = {"normal": "NORMAL (standard)", "deviant": "DEVIANT (violation)"}
N_PERM = 1000
RNG = np.random.default_rng(0)

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

    n0 = len(long)
    long = long[long["trial"] >= TEST_PHASE_MIN_TRIAL].reset_index(drop=True)
    print(f"Test-phase filter (trial >= {TEST_PHASE_MIN_TRIAL}): "
          f"removed {n0 - len(long)}, kept {len(long)}.")

    keep = dedup_runs(long[["animal", "date", "run", "sess"]])
    dropped = sorted(set(long["sess"].unique()) - keep)
    if dropped:
        print("Dedup (kept highest run) -> dropped:", dropped)
    long = long[long["sess"].isin(keep)].reset_index(drop=True)

    before_g = long["sess"].nunique()
    long = long[long["group"].isin(GROUP_ORDER)].reset_index(drop=True)
    print(f"Restricted to {GROUP_ORDER}: {before_g} -> {long['sess'].nunique()} sessions.")

    long["half"] = "mid"
    for (sess, cond), sub in long.groupby(["sess", "condition"]):
        order = sub.sort_values("trial")["trial"].values
        n = len(order)
        if SPLIT_MODE == "fixed":
            early, late = set(order[:N_EARLY_LATE]), set(order[-N_EARLY_LATE:])
            overlap = early & late          # session had < 2N trials
            early, late = early - overlap, late - overlap
        else:                                # median split
            h = n // 2
            early, late = set(order[:h]), set(order[n - h:])
        m = (long.sess == sess) & (long.condition == cond)
        long.loc[m, "half"] = long.loc[m, "trial"].map(
            lambda t: "early" if t in early else ("late" if t in late else "mid"))
    long = long[long["half"].isin(["early", "late"])].reset_index(drop=True)
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


def cluster_perm_sig(diff, taxis, n_perm=N_PERM, alpha=0.05, seed=1):
    diff = np.asarray(diff, float)
    n_sess, n_t = diff.shape
    sig = np.zeros(n_t, bool)
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


DEVIANT_PIPS = {"control_newsound": {"B", "C", "D"}}


def deviant_pips_for(group):
    return DEVIANT_PIPS.get(group, {"C"})


def shade_pips(ax, group=None):
    dev = (deviant_pips_for(group) if isinstance(group, str)
           else set().union(*[deviant_pips_for(g) for g in group])
           if group else {"C"})
    for label, (a, b) in PIP_WINDOWS.items():
        is_dev = label.split()[0] in dev
        ax.axvspan(a, b, color=("#F2C14E" if is_dev else "#000000"),
                   alpha=(0.18 if is_dev else 0.05), zorder=0)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    y0, y1 = ax.get_ylim()
    for label, (a, b) in PIP_WINDOWS.items():
        ax.text((a + b) / 2, y1 - 0.06 * (y1 - y0), label[0],
                ha="center", va="top", fontsize=8, fontweight="bold")


def mark_pips_thesis(ax, cond, group=None):
    """Thesis Fig-13 style: thin grey bars at pip boundaries; RED bar(s) at the
    deviant tone onset(s) on the deviant panel. For control_newsound the deviant
    spans B/C/D so B and D onsets are also marked red."""
    dev = deviant_pips_for(group) if group else {"C"}
    onsets = {"A": 0.0, "B": 0.25, "C": 0.50, "D": 0.75}
    for x in [0.0, 0.25, 0.50, 0.75, 1.00]:
        ax.axvline(x, color="0.6", lw=0.8, zorder=0)
    if cond == "deviant":
        for p in dev:
            ax.axvline(onsets[p], color="red", lw=1.6, zorder=1)
    y0, y1 = ax.get_ylim()
    for lab, x in zip("ABCD", [0.125, 0.375, 0.625, 0.875]):
        ax.text(x, y1 - 0.05 * (y1 - y0), lab, ha="center", va="top",
                fontsize=9, fontweight="bold")


def groups_present(df):
    return [g for g in GROUP_ORDER if g in set(df["group"])]


def build_counts(long):
    g = (long.groupby(["sess", "animal", "date", "group", "condition", "half"])
             ["trial"].nunique().unstack("half").fillna(0).astype(int).reset_index())
    for c in ["early", "late"]:
        if c not in g:
            g[c] = 0
    g = g.rename(columns={"early": "n_early", "late": "n_late"})
    need = N_EARLY_LATE if SPLIT_MODE == "fixed" else MIN_PER_HALF
    g["testable"] = (g["n_early"] >= need) & (g["n_late"] >= need)
    g["group"] = pd.Categorical(g["group"], categories=GROUP_ORDER, ordered=True)
    return g.sort_values(["condition", "group", "sess"])


def session_half_means(long, taxis):
    tcols = list(taxis)
    rows = []
    for (animal, date, sess, grp, cond, half), sub in long.groupby(
            ["animal", "date", "sess", "group", "condition", "half"]):
        mat = sub[tcols].to_numpy(float)
        rows.append({"animal": animal, "date": date, "sess": sess, "group": grp,
                     "condition": cond, "half": half, "n": len(sub),
                     "mean_trace": np.nanmean(mat, axis=0),
                     "max_resp": np.nanmean(win_max(mat, taxis, RESPONSE_WINDOW))})
    return pd.DataFrame(rows)


def plot_per_session(long, taxis, testable, cond):
    tcols = list(taxis)
    sc = long[long.condition == cond]
    meta = (sc[["animal", "date", "sess", "group"]]
            .drop_duplicates().sort_values(["animal", "date"]))
    for animal in sorted(sc["animal"].unique()):
        sa = meta[meta.animal == animal]
        n = len(sa)
        fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, sa.iterrows()):
            for half in ["early", "late"]:
                s = sc[(sc.sess == row.sess) & (sc.half == half)]
                if s.empty:
                    continue
                mat = s[tcols].to_numpy(float)
                mu = np.nanmean(mat, 0); se = sem(mat, 0)
                ax.plot(taxis, mu, color=HALF_COLORS[half], lw=1.6,
                        label=f"{half} (n={len(s)})")
                ax.fill_between(taxis, mu - se, mu + se,
                                color=HALF_COLORS[half], alpha=0.25, lw=0)
            ok = row.sess in testable
            ax.set_title(f"{row.date}\n[{row.group}]" + ("" if ok else " too few"),
                         fontsize=8, color=("black" if ok else "red"))
            ax.axhline(0, color="k", lw=0.5); shade_pips(ax, row.group)
            ax.set_xlabel("time from pattern onset (s)", fontsize=8)
            ax.legend(fontsize=5.5, loc="upper left", frameon=False)
        axes[0].set_ylabel(f"z-scored pupil\n({cond})")
        fig.suptitle(f"{animal} — {COND_TITLE[cond]} early vs late "
                     f"({SPLIT_MODE} split, test phase; red = too few)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        f = os.path.join(OUTDIR, f"A1_{cond}_early_late_{animal}.png")
        fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


def plot_thesis_replica(shm, taxis):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panel = {"normal": "A  normal", "deviant": "B  deviant"}
    for ax, cond in zip(axes, ["normal", "deviant"]):
        for half in ["early", "late"]:
            sub = shm[(shm.condition == cond) & (shm.half == half)]
            if sub.empty:
                continue
            tr = np.vstack(sub["mean_trace"].to_numpy())
            mu = np.nanmean(tr, 0); se = sem(tr, 0)
            ax.plot(taxis, mu, color=HALF_COLORS[half], lw=2,
                    label=f"{half} ({len(sub)} sess)")
            ax.fill_between(taxis, mu - se, mu + se,
                            color=HALF_COLORS[half], alpha=0.25, lw=0)
        ax.axhline(0, color="k", lw=0.5); ax.set_xlim(-1.0, 3.0)
        mark_pips_thesis(ax, cond)
        ax.set_title(panel[cond], fontsize=11, loc="left")
        ax.set_xlabel("time from pattern onset (s)")
        ax.legend(fontsize=9, frameon=False, loc="upper right")
    axes[0].set_ylabel("pupil dilation (z-scored)")
    n = N_EARLY_LATE if SPLIT_MODE == "fixed" else "half"
    fig.suptitle(f"Early vs late ({SPLIT_MODE} split, "
                 f"N={n}) of the test phase, all sessions\n"
                 "(red bar = deviant C tone; SEM across sessions)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    f = os.path.join(OUTDIR, "A2a_thesis_fig13_replica.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


def plot_grandavg_by_group(shm, taxis):
    gps = groups_present(shm)
    fig, axes = plt.subplots(2, len(gps), figsize=(4.2 * len(gps), 7),
                             sharey=True, sharex=True)
    if len(gps) == 1:
        axes = axes.reshape(2, 1)
    for r, cond in enumerate(["normal", "deviant"]):
        for c, grp in enumerate(gps):
            ax = axes[r, c]
            for half in ["early", "late"]:
                sub = shm[(shm.group == grp) & (shm.condition == cond) &
                          (shm.half == half)]
                if sub.empty:
                    continue
                tr = np.vstack(sub["mean_trace"].to_numpy())
                mu = np.nanmean(tr, 0); se = sem(tr, 0)
                ax.plot(taxis, mu, color=HALF_COLORS[half], lw=2,
                        label=f"{half} ({len(sub)})")
                ax.fill_between(taxis, mu - se, mu + se,
                                color=HALF_COLORS[half], alpha=0.25, lw=0)
            ax.axhline(0, color="k", lw=0.5); shade_pips(ax, grp)
            ax.legend(fontsize=6.5, loc="upper left", frameon=False)
            if r == 0:
                ax.set_title(grp, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{COND_TITLE[cond]}\nz-scored pupil")
            if r == 1:
                ax.set_xlabel("time from pattern onset (s)")
    fig.suptitle("Supplementary — early vs late broken down by group "
                 "(normal top, deviant bottom)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    f = os.path.join(OUTDIR, "A2b_grandavg_by_group.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


def plot_difference(shm, taxis):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    for ax, cond in zip(axes, ["normal", "deviant"]):
        for grp in groups_present(shm):
            piv = {h: {r.sess: r.mean_trace for _, r in
                       shm[(shm.group == grp) & (shm.condition == cond) &
                           (shm.half == h)].iterrows()} for h in ["early", "late"]}
            shared = sorted(set(piv["early"]) & set(piv["late"]))
            if not shared:
                continue
            diff = np.vstack([piv["late"][s] - piv["early"][s] for s in shared])
            mu = np.nanmean(diff, 0); se = sem(diff, 0)
            ax.plot(taxis, mu, color=GROUP_COLORS[grp], lw=2,
                    label=f"{grp} ({len(shared)})")
            ax.fill_between(taxis, mu - se, mu + se,
                            color=GROUP_COLORS[grp], alpha=0.15, lw=0)
            mask = (cluster_perm_sig(diff, taxis) if len(shared) >= 3
                    else np.zeros(len(taxis), bool))
            if mask.any():
                y0, y1 = ax.get_ylim()
                ax.fill_between(taxis, y0, y1, where=mask,
                                color=GROUP_COLORS[grp], alpha=0.12, lw=0, zorder=0)
        ax.axhline(0, color="k", lw=0.6); shade_pips(ax, groups_present(shm))
        ax.set_xlabel("time from pattern onset (s)")
        ax.set_title(COND_TITLE[cond], fontsize=11)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("Δ pupil (late − early), session-paired")
    fig.suptitle("Late − early difference (shaded = significant cluster, "
                 "permutation p<.05; negative = adaptation)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f = os.path.join(OUTDIR, "A3_late_minus_early_difference.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)


def plot_adaptation_index(shm, testable_by_cond):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    out = {}
    for ax, cond in zip(axes, ["normal", "deviant"]):
        piv = (shm[shm.condition == cond]
               .pivot_table(index=["sess", "animal", "group"], columns="half",
                            values="max_resp").reset_index())
        piv = piv.dropna(subset=["early", "late"])
        piv["adapt"] = piv["late"] - piv["early"]
        out[cond] = piv
        pos = 0; xt, xl = [], []
        for grp in [g for g in GROUP_ORDER if g in set(piv["group"])]:
            sub = piv[piv.group == grp].sort_values("animal")
            for _, r in sub.iterrows():
                pos += 1
                ok = r.sess in testable_by_cond[cond]
                ax.bar(pos, r.adapt, color=GROUP_COLORS[grp],
                       edgecolor=("k" if ok else "red"),
                       linewidth=(0.5 if ok else 1.6), alpha=0.85)
                xt.append(pos); xl.append(f"{r.animal}\n{r.sess.split('_')[1]}")
            pos += 0.8
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=4.5)
        ax.set_title(f"{COND_TITLE[cond]}\nnegative = response shrinks late",
                     fontsize=10)
    axes[0].set_ylabel("adaptation index: late − early max pupil")
    fig.suptitle("Per-session adaptation index (red edge = too few trials to trust)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f = os.path.join(OUTDIR, "A4_adaptation_index.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("saved", f)
    return out


def run_stats(shm, long, taxis, testable_by_cond, adapt, lines):
    def W(s=""): lines.append(s); print(s)
    W("=" * 78)
    W(f"ADAPTATION — early vs late ({SPLIT_MODE} split, "
      f"N={N_EARLY_LATE if SPLIT_MODE=='fixed' else 'half'}), TEST PHASE")
    W("=" * 78)
    W(f"Animals: {sorted(long['animal'].unique())} | "
      f"response window {RESPONSE_WINDOW[0]}–{RESPONSE_WINDOW[1]} s")

    for cond in ["normal", "deviant"]:
        W(f"\n#### {COND_TITLE[cond]} ####")
        tset = testable_by_cond[cond]
        piv = adapt[cond]
        piv = piv[piv.sess.isin(tset)]
        W(f"  testable sessions: {len(piv)}")
        if len(piv) >= 3:
            try:
                _, wp = stats.wilcoxon(piv["late"], piv["early"])
            except ValueError:
                wp = np.nan
            _, tp = stats.ttest_rel(piv["late"], piv["early"])
            W(f"  OVERALL: mean(late-early)={piv['adapt'].mean():+.3f} | "
              f"Wilcoxon p={wp:.4f} | paired t p={tp:.4f}  (negative=adaptation)")
        for grp in [g for g in GROUP_ORDER if g in set(piv["group"])]:
            s = piv[piv.group == grp]
            if len(s) >= 3:
                try:
                    _, wp = stats.wilcoxon(s["late"], s["early"])
                except ValueError:
                    wp = np.nan
                _, tp = stats.ttest_rel(s["late"], s["early"])
                W(f"    {grp:18s}: n={len(s):2d} mean(late-early)={s['adapt'].mean():+.3f} "
                  f"Wilcoxon p={wp:.4f} paired t p={tp:.4f}")
            else:
                m = s['adapt'].mean() if len(s) else float('nan')
                W(f"    {grp:18s}: n={len(s):2d} — too few (mean Δ={m:+.3f})")

        gps = [s["adapt"].values for _, s in piv.groupby("group") if len(s) >= 2]
        if len(gps) >= 2:
            h, p = stats.kruskal(*gps)
            W(f"    Kruskal-Wallis across groups: H={h:.3f}, p={p:.4f}")

        tcols = list(taxis)
        base = long[long.condition == cond].copy()
        mat = base[tcols].to_numpy(float)
        m = (taxis > RESPONSE_WINDOW[0]) & (taxis <= RESPONSE_WINDOW[1])
        with np.errstate(all="ignore"):
            base = base.assign(max_resp=np.nanmax(mat[:, m], axis=1))
        df = base[base.sess.isin(tset)].dropna(subset=["max_resp"]).copy()
        df["half"] = pd.Categorical(df["half"], categories=["early", "late"])
        if df.sess.nunique() >= 3:
            try:
                mf = smf.mixedlm("max_resp ~ half", df, groups=df["animal"],
                                 vc_formula={"sess": "0 + C(sess)"},
                                 re_formula="1").fit(method="lbfgs", maxiter=300,
                                                     disp=False)
                for k in mf.params.index:
                    if ("half" in k or "Intercept" in k) and "Var" not in k:
                        W(f"    MM {k:20s} beta={mf.params[k]:+.4f} "
                          f"p={mf.pvalues[k]:.4f}")
            except Exception as e:
                W(f"    mixed model failed ({e})")

    W("\nNotes:")
    W(" - Early/late are split PER CONDITION within each session, test phase only.")
    if SPLIT_MODE == "fixed":
        W(f" - Fixed N={N_EARLY_LATE} needs >= {2*N_EARLY_LATE} trials/condition;")
        W("   sparser sessions are flagged and excluded from that condition's stats.")
    W(" - Negative (late-early) = adaptation. Positive = response grows.")


def main():
    lines = []
    long = load_long()
    taxis = time_axis(long)

    counts = build_counts(long)
    counts.to_csv(os.path.join(OUTDIR, "A0_early_late_counts.csv"), index=False)
    testable_by_cond = {c: set(counts[(counts.condition == c) & counts.testable]["sess"])
                        for c in ["normal", "deviant"]}
    print("Testable — normal:", len(testable_by_cond["normal"]),
          "| deviant:", len(testable_by_cond["deviant"]))

    for cond in ["deviant", "normal"]:
        plot_per_session(long, taxis, testable_by_cond[cond], cond)
    shm = session_half_means(long, taxis)
    shm.drop(columns=["mean_trace"]).to_csv(
        os.path.join(OUTDIR, "A_per_session_half_summaries.csv"), index=False)

    plot_thesis_replica(shm, taxis)
    plot_grandavg_by_group(shm, taxis)
    plot_difference(shm, taxis)
    adapt = plot_adaptation_index(shm, testable_by_cond)
    for cond, dfc in adapt.items():
        dfc.to_csv(os.path.join(OUTDIR, f"A_adaptation_index_{cond}.csv"),
                   index=False)

    run_stats(shm, long, taxis, testable_by_cond, adapt, lines)
    with open(os.path.join(OUTDIR, "adaptation_statistics.txt"), "w") as fh:
        fh.write("\n".join(lines))
    print("\nDONE. Outputs in", OUTDIR)


if __name__ == "__main__":
    main()
