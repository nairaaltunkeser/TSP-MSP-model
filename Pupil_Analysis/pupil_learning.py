#!/usr/bin/env python3
# ============================================================================
#  LEARNING ANALYSIS — does the deviant response grow across sessions?
#
#  Question: do the animals slowly learn the sequence across sessions, i.e.
#  does the pupil response to the DEVIANT (relative to NORMAL) increase with
#  session number?
#
#  Why the JUNE BLOCK ONLY:
#  The full timeline confounds "sessions over time" with a change of sound set
#  in July. To isolate learning we restrict to the original single-sound block
#  (dates 260603-260607), which uses one fixed sound set throughout. Only
#  NA01/NA02/NA03 have sessions here (NA04/NA05 start in July), so the learning
#  test is necessarily those three animals.
#
#  For each animal we take the per-session scalar pupil response (max and mean
#  in the 0.5-2.0 s window) for normal and deviant, and the deviant-normal
#  difference, then fit an ordinary least-squares LINE against session order
#  (0,1,2,...). A positive, significant slope on the deviant-normal difference
#  = the violation response grows across sessions (consistent with learning).
#
#  Because each animal has only 3-5 sessions, per-animal slopes are noisy; we
#  therefore ALSO fit a linear mixed model across animals:
#       resp ~ session_index * condition + (1 | animal)
#  and report the session:deviant interaction (the group-level learning term),
#  which pools the animals and is the more trustworthy test.
#
#  Test phase only (trial >= 101). Output -> outputs_learning/
#     L1_learning_slopes_max.png / _mean.png    per-animal lines + fitted slope
#     L2_deviant_minus_normal_slope.png         the key contrast, both metrics
#     learning_statistics.txt                   per-animal slopes + mixed model
#     learning_session_values.csv
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

# ----------------------------------------------------------------------------
PUPIL_H5 = "EC_opto_pupil.h5"
OUTDIR   = "outputs_learning"

# Single-sound original block only, to isolate learning from the July sound change
JUNE_BLOCK_DATES = ["260603", "260604", "260605", "260606", "260607"]
TEST_PHASE_MIN_TRIAL = 101
RESPONSE_WINDOW = (0.50, 2.00)

COND_COLORS = {"normal": "#4C72B0", "deviant": "#C44E52"}
DIFF_COLOR = "#6A3D9A"

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(_HERE, OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)


def resolve_data_file(fname):
    direct = os.path.join(_HERE, fname)
    if os.path.exists(direct):
        return direct
    entries = sorted(os.listdir(_HERE))
    for e in entries:
        if e.lower().replace(" ", "") == fname.lower().replace(" ", ""):
            print(f"NOTE: using '{e}' for requested '{fname}'.")
            return os.path.join(_HERE, e)
    h5 = [e for e in entries if e.lower().endswith((".h5", ".hdf5"))]
    raise FileNotFoundError(
        f"Could not find '{fname}' in:\n  {_HERE}\n\n"
        + ("HDF5 files present:\n  " + "\n  ".join(repr(e) for e in h5)
           if h5 else "No .h5 files found here."))


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
        vals = df.reset_index(drop=True)
        vals.columns = [float(c) for c in vals.columns]
        frames.append(pd.concat([meta.reset_index(drop=True), vals], axis=1))
    long = pd.concat(frames, ignore_index=True)
    long = long[long["trial"] >= TEST_PHASE_MIN_TRIAL].reset_index(drop=True)
    keep = dedup_runs(long[["animal", "date", "run", "sess"]])
    long = long[long["sess"].isin(keep)].reset_index(drop=True)
    long = long[long["date"].isin(JUNE_BLOCK_DATES)].reset_index(drop=True)
    print("June-block sessions per animal:")
    print(long.groupby("animal")["date"].apply(lambda x: sorted(set(x))).to_string())
    return long


def time_axis(long):
    return np.array(sorted(c for c in long.columns if isinstance(c, float)))


def build_table(long, taxis):
    tcols = list(taxis)
    m = (taxis > RESPONSE_WINDOW[0]) & (taxis <= RESPONSE_WINDOW[1])
    rows = []
    for (animal, date, sess, cond), sub in long.groupby(
            ["animal", "date", "sess", "condition"]):
        mat = sub[tcols].to_numpy(float)
        with np.errstate(all="ignore"):
            rows.append({"animal": animal, "date": date, "sess": sess,
                         "condition": cond, "n_trials": len(sub),
                         "max_resp": np.nanmean(np.nanmax(mat[:, m], axis=1)),
                         "mean_resp": np.nanmean(np.nanmean(mat[:, m], axis=1))})
    df = pd.DataFrame(rows)
    # session index per animal (0,1,2,... in date order)
    df = df.sort_values(["animal", "date"])
    df["sidx"] = df.groupby("animal")["date"].transform(
        lambda s: pd.Series(pd.factorize(s)[0], index=s.index))
    return df


def ols_slope(x, y):
    """Return slope, p-value, intercept for y ~ x (needs >= 3 points)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~np.isnan(y)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    res = stats.linregress(x[ok], y[ok])
    return res.slope, res.pvalue, res.intercept


# ----------------------------------------------------------------------------
def plot_per_animal(df, metric, ylabel, fname, lines):
    animals = sorted(df["animal"].unique())
    fig, axes = plt.subplots(1, len(animals), figsize=(4.4 * len(animals), 3.8),
                             sharey=True, squeeze=False)
    axes = axes[0]
    lines.append(f"\n=== per-animal OLS slopes vs session index — {ylabel} ===")
    for ax, animal in zip(axes, animals):
        da = df[df.animal == animal]
        dates = sorted(da["date"].unique())
        for cond in ["normal", "deviant"]:
            dc = da[da.condition == cond].sort_values("sidx")
            if dc.empty:
                continue
            ax.plot(dc["sidx"], dc[metric], marker="o", ms=6,
                    color=COND_COLORS[cond], lw=1.8, label=cond)
            sl, pv, ic = ols_slope(dc["sidx"], dc[metric])
            if not np.isnan(sl):
                xs = np.array([dc["sidx"].min(), dc["sidx"].max()])
                ax.plot(xs, ic + sl * xs, color=COND_COLORS[cond], lw=1,
                        ls="--", alpha=0.7)
                lines.append(f"  {animal} {cond:8s}: slope={sl:+.3f} "
                             f"p={pv:.3f} (n={dc[metric].notna().sum()})")
        # deviant - normal difference slope
        piv = da.pivot_table(index="sidx", columns="condition", values=metric)
        if {"normal", "deviant"}.issubset(piv.columns):
            diff = (piv["deviant"] - piv["normal"]).dropna()
            sl, pv, ic = ols_slope(diff.index.values, diff.values)
            tag = ""
            if not np.isnan(sl):
                tag = f"  dev-nor slope={sl:+.3f} p={pv:.3f}"
                lines.append(f"  {animal} DEV-NOR : slope={sl:+.3f} p={pv:.3f}")
        ax.set_title(f"{animal}{tag}", fontsize=9)
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, fontsize=7, ha="right")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("session order")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{ylabel} across June-block sessions (single sound set)\n"
                 "dashed = fitted OLS slope; a rising deviant line = learning",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f = os.path.join(OUTDIR, fname)
    fig.savefig(f, dpi=150); plt.close(fig); print("saved", f)


def plot_diff_slopes(df, lines):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, (metric, lab) in zip(axes, [("max_resp", "max"),
                                        ("mean_resp", "mean")]):
        for animal in sorted(df["animal"].unique()):
            da = df[df.animal == animal]
            piv = da.pivot_table(index="sidx", columns="condition", values=metric)
            if not {"normal", "deviant"}.issubset(piv.columns):
                continue
            diff = (piv["deviant"] - piv["normal"]).dropna()
            ax.plot(diff.index, diff.values, marker="o", ms=6, lw=1.8,
                    label=animal)
            sl, pv, ic = ols_slope(diff.index.values, diff.values)
            if not np.isnan(sl):
                xs = np.array([diff.index.min(), diff.index.max()])
                ax.plot(xs, ic + sl * xs, ls="--", lw=1, alpha=0.6,
                        color=ax.get_lines()[-1].get_color())
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"deviant − normal ({lab} pupil)", fontsize=10)
        ax.set_xlabel("session order")
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("Δ pupil (deviant − normal)")
    fig.suptitle("Learning contrast: does deviant − normal grow across sessions?\n"
                 "(June block, single sound set; upward slope = learning)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    f = os.path.join(OUTDIR, "L2_deviant_minus_normal_slope.png")
    fig.savefig(f, dpi=150); plt.close(fig); print("saved", f)


def mixed_model(df, metric, lab, lines):
    d = df.dropna(subset=[metric]).copy()
    d["condition"] = pd.Categorical(d["condition"],
                                    categories=["normal", "deviant"])
    lines.append(f"\n=== mixed model — {lab} pupil ===")
    lines.append(f"    {metric} ~ sidx * condition + (1 | animal)")
    lines.append("    the sidx:condition[deviant] term is the group-level "
                 "learning slope.")
    try:
        mf = smf.mixedlm(f"{metric} ~ sidx * condition", d,
                         groups=d["animal"]).fit(method="lbfgs", disp=False)
        for k in mf.params.index:
            if k == "Group Var":
                continue
            star = "  <-- learning term" if "sidx:" in k else ""
            lines.append(f"      {k:32s} beta={mf.params[k]:+.4f} "
                         f"p={mf.pvalues[k]:.4f}{star}")
        lines.append(f"      (converged={mf.converged})")
    except Exception as e:
        lines.append(f"      mixed model failed ({e})")


def main():
    lines = ["=" * 74,
             "LEARNING ANALYSIS — deviant response across sessions (June block)",
             "=" * 74,
             "Restricted to single-sound block 260603-260607 to isolate learning",
             "from the July sound change. NA01/NA02/NA03 only (NA04/05 start July)."]
    long = load_long()
    taxis = time_axis(long)
    df = build_table(long, taxis)
    df.to_csv(os.path.join(OUTDIR, "learning_session_values.csv"), index=False)

    lines.append("\nSessions per animal: "
                 + df.groupby("animal")["sidx"].max().add(1).to_dict().__repr__())

    plot_per_animal(df, "max_resp", "max pupil (0.5-2.0 s)",
                    "L1_learning_slopes_max.png", lines)
    plot_per_animal(df, "mean_resp", "mean pupil (0.5-2.0 s)",
                    "L1_learning_slopes_mean.png", lines)
    plot_diff_slopes(df, lines)

    mixed_model(df, "max_resp", "max", lines)
    mixed_model(df, "mean_resp", "mean", lines)

    lines.append("\nHow to read this:")
    lines.append(" - Per-animal slopes are from only 3-5 sessions, so individual")
    lines.append("   p-values are weak; the mixed-model interaction is the main test.")
    lines.append(" - Positive sidx:deviant slope + p<.05 => deviant response grows")
    lines.append("   across sessions relative to normal = evidence of learning.")
    lines.append(" - A flat/non-significant term => no across-session learning in")
    lines.append("   this block (the eyeballed July rise is then the sound change,")
    lines.append("   not gradual learning).")

    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(OUTDIR, "learning_statistics.txt"), "w") as fh:
        fh.write(txt)
    print("\nDONE. Outputs in", OUTDIR)


if __name__ == "__main__":
    main()
