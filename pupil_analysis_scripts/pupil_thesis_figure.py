#!/usr/bin/env python3
# Generates the thesis figure comparing control and opto pupil responses.
# Shows normal vs deviant responses and the deviant-minus-normal interaction.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

PUPIL_H5 = "EC_opto_pupil.h5"
OUTDIR   = "outputs_thesis_figure"

KEEP_GROUPS = ["control", "opto"]
TEST_PHASE_MIN_TRIAL = 101
RESPONSE_WINDOW = (0.50, 2.00)

NA45_ANIMALS       = {"NA04", "NA05"}
NA45_OPTO_DATES    = {"260720", "260721", "260722"}   # light-on  = opto
NA45_CONTROL_DATES = {"260723", "260724", "260725"}   # lights off = control
CONTROL_DATES          = {"260606", "260607", "260717", "260720", "260721", "260722"}
NA03_OPTO_DATES        = {"260606", "260607"}
OPTOLATE_DATES         = {"260618", "260619", "260620"}
CONTROL_NEWSOUND_DATES = {"260710", "260713", "260715"}

COND_COLORS  = {"normal": "#7BA7D0", "deviant": "#D48A8A"}
COND_EDGE    = {"normal": "#2C5F8A", "deviant": "#A83232"}
GROUP_COLORS = {"control": "#555555", "opto": "#2CA02C"}
RNG = np.random.default_rng(0)

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
            print(f"NOTE: using '{e}' for '{fname}'.")
            return os.path.join(_HERE, e)
    h5 = [e for e in entries if e.lower().endswith((".h5", ".hdf5"))]
    raise FileNotFoundError(
        f"Could not find '{fname}' in:\n  {_HERE}\n\n"
        + ("HDF5 files present:\n  " + "\n  ".join(repr(e) for e in h5)
           if h5 else "No .h5 files found here."))


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
    long = long[long["group"].isin(KEEP_GROUPS)].reset_index(drop=True)
    return long


def time_axis(long):
    return np.array(sorted(c for c in long.columns if isinstance(c, float)))


def win_mean(mat, taxis):
    m = (taxis > RESPONSE_WINDOW[0]) & (taxis <= RESPONSE_WINDOW[1])
    return np.nanmean(mat[:, m], axis=1)


def sem(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan


def p_to_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def main():
    L = []
    def W(s=""): L.append(s); print(s)

    long = load_long()
    taxis = time_axis(long)
    tcols = list(taxis)

    rows = []
    for (sess, animal, grp, cond), sub in long.groupby(
            ["sess", "animal", "group", "condition"]):
        mat = sub[tcols].to_numpy(float)
        rows.append({"sess": sess, "animal": animal, "group": grp,
                     "condition": cond, "val": np.nanmean(win_mean(mat, taxis))})
    sess_df = pd.DataFrame(rows)
    wide = sess_df.pivot_table(index=["sess", "animal", "group"],
                               columns="condition", values="val").reset_index()
    wide = wide.dropna(subset=["normal", "deviant"])
    wide["diff"] = wide["deviant"] - wide["normal"]

    W("=" * 66)
    W("THESIS FIGURE STATS — control vs opto, mean pupil 0.5-2.0 s")
    W("=" * 66)
    within_p = {}
    for g in KEEP_GROUPS:
        s = wide[wide.group == g]
        try:
            _, wp = stats.wilcoxon(s["deviant"], s["normal"])
        except ValueError:
            wp = np.nan
        within_p[g] = wp
        W(f"{g:8s}: n={len(s)}  normal={s['normal'].mean():.3f}  "
          f"deviant={s['deviant'].mean():.3f}  Δ={s['diff'].mean():+.3f}  "
          f"paired Wilcoxon p={wp:.4f}")
    a = wide[wide.group == "control"]["diff"].values
    b = wide[wide.group == "opto"]["diff"].values
    u, mw_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    W(f"Between groups (Mann-Whitney on Δ): U={u:.1f}, p={mw_p:.4f}")

    trials = long[["animal", "sess", "group", "condition"]].copy()
    trials["val"] = win_mean(long[tcols].to_numpy(float), taxis)
    trials = trials.dropna(subset=["val"])
    trials["condition"] = pd.Categorical(trials["condition"],
                                         categories=["normal", "deviant"])
    trials["group"] = pd.Categorical(trials["group"],
                                     categories=["control", "opto"])
    inter_p = np.nan
    try:
        mf = smf.mixedlm("val ~ condition * group", trials,
                         groups=trials["animal"]).fit(method="lbfgs", disp=False)
        inter_key = "condition[T.deviant]:group[T.opto]"
        inter_p = mf.pvalues.get(inter_key, np.nan)
        inter_b = mf.params.get(inter_key, np.nan)
        W(f"Mixed model interaction condition×group: "
          f"beta={inter_b:+.4f}, p={inter_p:.4f}  (KEY TEST)")
    except Exception as e:
        W(f"mixed model failed ({e})")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    xbase = {"control": 0, "opto": 1.6}
    offset = {"normal": -0.32, "deviant": 0.32}
    barw = 0.58
    for g in KEEP_GROUPS:
        for cond in ["normal", "deviant"]:
            vals = wide[wide.group == g][cond].values
            x = xbase[g] + offset[cond]
            axA.bar(x, np.mean(vals), barw, color=COND_COLORS[cond],
                    edgecolor=COND_EDGE[cond], linewidth=1.3, zorder=2,
                    label=cond if g == "control" else None)
            axA.errorbar(x, np.mean(vals), yerr=sem(vals), color="k",
                         lw=1.3, capsize=4, zorder=4)
        s = wide[wide.group == g]
        xn = xbase[g] + offset["normal"] + RNG.uniform(-0.08, 0.08, len(s))
        xd = xbase[g] + offset["deviant"] + RNG.uniform(-0.08, 0.08, len(s))
        for i in range(len(s)):
            axA.plot([xn[i], xd[i]],
                     [s["normal"].values[i], s["deviant"].values[i]],
                     color="0.6", lw=0.6, alpha=0.6, zorder=3)
        axA.scatter(xn, s["normal"].values, s=22, color=COND_COLORS["normal"],
                    edgecolor="k", linewidth=0.4, zorder=5)
        axA.scatter(xd, s["deviant"].values, s=22, color=COND_COLORS["deviant"],
                    edgecolor="k", linewidth=0.4, zorder=5)
        top = max(wide[wide.group == g]["normal"].max(),
                  wide[wide.group == g]["deviant"].max())
        ybr = top + 0.12
        x1, x2 = xbase[g] + offset["normal"], xbase[g] + offset["deviant"]
        axA.plot([x1, x1, x2, x2], [ybr, ybr + 0.04, ybr + 0.04, ybr],
                 color="k", lw=1.2)
        axA.text((x1 + x2) / 2, ybr + 0.05, p_to_stars(within_p[g]),
                 ha="center", va="bottom", fontsize=12, fontweight="bold")

    axA.set_xticks(list(xbase.values()))
    axA.set_xticklabels([g for g in KEEP_GROUPS], fontsize=12)
    axA.set_ylabel("mean pupil (0.5–2.0 s window, z)", fontsize=11)
    axA.axhline(0, color="k", lw=0.6)
    yl = axA.get_ylim()
    axA.set_ylim(yl[0], yl[1] + 0.18 * (yl[1] - yl[0]))
    axA.set_title("A  Pupil response (within-group paired Wilcoxon test)",
                  fontsize=12, loc="left")
    from matplotlib.patches import Patch
    axA.legend(handles=[Patch(facecolor=COND_COLORS["normal"],
                              edgecolor=COND_EDGE["normal"], label="normal"),
                        Patch(facecolor=COND_COLORS["deviant"],
                              edgecolor=COND_EDGE["deviant"], label="deviant")],
               fontsize=10, frameon=False, loc="upper right")

    for i, g in enumerate(KEEP_GROUPS):
        d = wide[wide.group == g]["diff"].values
        axB.bar(i, np.mean(d), 0.6, color=GROUP_COLORS[g], alpha=0.35,
                edgecolor=GROUP_COLORS[g], linewidth=1.5, zorder=2)
        axB.errorbar(i, np.mean(d), yerr=sem(d), color="k", lw=1.4,
                     capsize=5, zorder=4)
        xj = i + RNG.uniform(-0.13, 0.13, len(d))
        axB.scatter(xj, d, s=32, color=GROUP_COLORS[g], edgecolor="k",
                    linewidth=0.4, zorder=5)
    axB.axhline(0, color="k", lw=0.8)
    axB.set_xticks(range(len(KEEP_GROUPS)))
    axB.set_xticklabels(KEEP_GROUPS, fontsize=12)
    axB.set_ylabel("deviant − normal  (Δ mean pupil)", fontsize=11)
    axB.set_title("B  Deviant surprise response (Linear Mixed Model)",
                  fontsize=12, loc="left")

    ymax = wide["diff"].max()
    ybr = ymax + 0.12
    axB.plot([0, 0, 1, 1], [ybr, ybr + 0.05, ybr + 0.05, ybr], color="k", lw=1.3)
    lab = (f"interaction p = {inter_p:.3f} {p_to_stars(inter_p)}"
           if inter_p == inter_p else "interaction n/a")
    axB.text(0.5, ybr + 0.06, lab, ha="center", va="bottom",
             fontsize=11, fontweight="bold")
    ylb = axB.get_ylim()
    axB.set_ylim(ylb[0], max(ylb[1], ybr + 0.22))

    fig.suptitle("Opto suppresses the deviant-evoked pupil surprise response",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["png", "pdf"]:
        f = os.path.join(OUTDIR, f"FIG_opto_deviant_interaction.{ext}")
        fig.savefig(f, dpi=200, bbox_inches="tight")
        print("saved", f)
    plt.close(fig)

    W("\nStars: * p<.05, ** p<.01, *** p<.001, n.s. otherwise.")
    W("Panel A brackets = within-group paired Wilcoxon (deviant vs normal).")
    W("Panel B bracket  = condition×group interaction (mixed model) = the key")
    W("test of whether opto changes the deviant response.")
    with open(os.path.join(OUTDIR, "thesis_figure_stats.txt"), "w") as fh:
        fh.write("\n".join(L))
    print("DONE. Outputs in", OUTDIR)


if __name__ == "__main__":
    main()
