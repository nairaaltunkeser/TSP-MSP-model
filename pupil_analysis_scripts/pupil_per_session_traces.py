#!/usr/bin/env python3
# Per-session pupil traces for control and opto sessions.
# Plots normal and deviant mean responses separately for each session.

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUPIL_H5 = "EC_opto_pupil.h5"
OUTDIR   = "outputs_per_session_traces"

KEEP_GROUPS = ["control", "opto"]
TEST_PHASE_MIN_TRIAL = 101

NA45_ANIMALS       = {"NA04", "NA05"}
NA45_OPTO_DATES    = {"260720", "260721", "260722"}   # light-on  = opto
NA45_CONTROL_DATES = {"260723", "260724", "260725"}   # lights off = control
CONTROL_DATES          = {"260606", "260607", "260717", "260720", "260721", "260722"}
NA03_OPTO_DATES        = {"260606", "260607"}
OPTOLATE_DATES         = {"260618", "260619", "260620"}
CONTROL_NEWSOUND_DATES = {"260710", "260713", "260715"}

PIP_WINDOWS = {"A": (0.00, 0.25), "B": (0.25, 0.50),
               "C": (0.50, 0.75), "D": (0.75, 1.00)}
COND_COLORS = {"normal": "#4C72B0", "deviant": "#C44E52"}
GROUP_COLORS = {"control": "#7F7F7F", "opto": "#2CA02C"}

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


def sem(a, axis=0):
    a = np.asarray(a, float)
    n = np.sum(~np.isnan(a), axis=axis)
    return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))


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
    print(f"Sessions kept ({KEEP_GROUPS}): {long['sess'].nunique()}")
    return long


def time_axis(long):
    return np.array(sorted(c for c in long.columns if isinstance(c, float)))


DEVIANT_PIPS = {"control_newsound": {"B", "C", "D"}}


def shade_pips(ax, group=None):
    dev = DEVIANT_PIPS.get(group, {"C"})
    for lab, (a, b) in PIP_WINDOWS.items():
        is_dev = lab in dev
        ax.axvspan(a, b, color=("#F2C14E" if is_dev else "#000000"),
                   alpha=(0.18 if is_dev else 0.05), zorder=0)
    ax.axvline(0, color="k", lw=0.6, ls=":")


def main():
    long = load_long()
    taxis = time_axis(long)
    tcols = list(taxis)

    sess_meta = (long[["sess", "animal", "date", "group"]].drop_duplicates())
    sess_meta["gorder"] = sess_meta["group"].map(
        {g: i for i, g in enumerate(KEEP_GROUPS)})
    sess_meta = sess_meta.sort_values(["gorder", "animal", "date"])
    sessions = sess_meta["sess"].tolist()

    n = len(sessions)
    ncol = 6
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.5 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    ymin, ymax = 0.0, 0.0
    per_sess = {}
    for sess in sessions:
        d = {}
        for cond in ["normal", "deviant"]:
            sub = long[(long.sess == sess) & (long.condition == cond)]
            if sub.empty:
                continue
            mat = sub[tcols].to_numpy(float)
            mu = np.nanmean(mat, 0); se = sem(mat, 0)
            d[cond] = (mu, se, len(sub))
            ymin = min(ymin, np.nanmin(mu - se))
            ymax = max(ymax, np.nanmax(mu + se))
        per_sess[sess] = d
    pad = 0.05 * (ymax - ymin)
    ylim = (ymin - pad, ymax + pad)

    for ax, sess in zip(axes, sessions):
        row = sess_meta[sess_meta.sess == sess].iloc[0]
        d = per_sess[sess]
        for cond in ["normal", "deviant"]:
            if cond not in d:
                continue
            mu, se, ntr = d[cond]
            ax.plot(taxis, mu, color=COND_COLORS[cond], lw=1.5,
                    label=f"{cond} (n={ntr})")
            ax.fill_between(taxis, mu - se, mu + se,
                            color=COND_COLORS[cond], alpha=0.22, lw=0)
        ax.set_ylim(ylim)
        ax.axhline(0, color="k", lw=0.4)
        shade_pips(ax, row.group)
        ax.set_title(f"{row.animal} {row.date}", fontsize=8.5,
                     color=GROUP_COLORS[row.group])
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.5, loc="upper left", frameon=False)

    for ax in axes[len(sessions):]:
        ax.axis("off")

    fig.supxlabel("time from pattern onset (s)", fontsize=10)
    fig.supylabel("z-scored pupil (baseline-corrected)", fontsize=10)
    fig.suptitle("Mean pupil response per session — normal vs deviant "
                 "(grey titles = control, green = opto; yellow band = deviant C tone)",
                 fontsize=12)
    fig.tight_layout(rect=[0.01, 0.02, 1, 0.96])
    f = os.path.join(OUTDIR, "S1_per_session_normal_vs_deviant.png")
    fig.savefig(f, dpi=150); plt.close(fig); print("saved", f)
    print("DONE. Outputs in", OUTDIR)


if __name__ == "__main__":
    main()
