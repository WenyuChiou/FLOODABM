# -*- coding: utf-8 -*-
"""
Income-normalized AEP curve: per-HH annual loss expressed as % of the tenure's
ACS median household income, so Homeowner and Renter sit on the SAME axis.

Pipeline mirrors Fig6 (e)/(f): 50-run MC, pool 50 runs x 12 years (drop 2011
warm-up), per (run,year) divide aggregate loss by that (run,year) household
count -> per-HH; then divide by the tenure's ACS median income; sort -> EP.
"""
import sys, glob, re
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MC_ROOT = Path(r"C:\FLOODABM_mc50_v2")
N_RUNS = 50
OUT = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\validation")
FIG_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\Figure")

# ACS county-weighted median household income by tenure (from census_reporter_income.py)
INCOME = {"owner": 148097.0, "renter": 63699.0}

C_OWN, C_REN = "#1a6faf", "#1f7a3c"   # (legacy, unused below)
# Match the main AEP figure (Fig6 e/f): GUL = red, Actual (uninsured) loss = blue
C_GUL, C_ACT = "#d62828", "#0077b6"


def run_path(run_id):
    return MC_ROOT / "baseline" / f"run_{run_id:02d}" / "baseline" / "baseline"


def load_run(rid):
    rows = []
    for f in sorted(glob.glob(str(run_path(rid) / "finance" / "finance_tract_*.csv"))):
        m = re.search(r"tract_(\d{4})\.csv", f)
        if not m:
            continue
        df = pd.read_csv(f, dtype={"tract_geoid": str})
        df["year"] = int(m.group(1))
        rows.append(df)
    if not rows:
        return None
    fin = pd.concat(rows, ignore_index=True)
    g = fin.groupby("year").agg(
        owner_gross=("owner_gross_total_usd", "sum"),
        renter_gross=("renter_gross_total_usd", "sum"),
        owner_payout=("owner_payout_total_usd", "sum"),
        renter_payout=("renter_payout_total_usd", "sum"),
        owner_hh=("owner_households", "sum"),
        renter_hh=("renter_households", "sum"),
    ).reset_index().fillna(0).sort_values("year")
    return g


print("loading 50 MC runs (baseline)...")
frames = [load_run(r) for r in range(1, N_RUNS + 1)]
frames = [f for f in frames if f is not None]
print(f"  loaded {len(frames)} runs")
cols = ["owner_gross", "renter_gross", "owner_payout", "renter_payout", "owner_hh", "renter_hh"]
D = {c: np.stack([f[c].values.astype(float) for f in frames], axis=0) for c in cols}  # (runs, years)

# per-HH, drop 2011 (col 0), divide by income -> % of income
def ratio(group):
    hh = np.maximum(D[f"{group}_hh"][:, 1:], 1.0)
    gul_ph = D[f"{group}_gross"][:, 1:] / hh
    loss_ph = (D[f"{group}_gross"][:, 1:] - D[f"{group}_payout"][:, 1:]) / hh
    inc = INCOME[group]
    return (np.sort(gul_ph.ravel()) / inc * 100.0,
            np.sort(loss_ph.ravel()) / inc * 100.0)


o_gul, o_loss = ratio("owner")
r_gul, r_loss = ratio("renter")
n = len(o_gul)
ep = 1.0 - np.arange(1, n + 1) / n

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 14,
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linestyle": "--",
})
# ONE axis (both tenures share the same unit: % of income). Colours match Fig6 e/f:
# GUL = red, Actual (uninsured) loss = blue; Homeowner = solid, Renter = dashed.
# No title; the covered-%-of-GUL is already in Fig6 and is NOT repeated here — this
# figure adds only the income dimension (the gap as a share of income).
from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(9.0, 6.2))
ax.plot(ep, o_gul,  color=C_GUL, ls="-",  lw=2.4, zorder=4)
ax.plot(ep, o_loss, color=C_ACT, ls="-",  lw=2.4, zorder=4)
ax.plot(ep, r_gul,  color=C_GUL, ls="--", lw=2.0, zorder=3)
ax.plot(ep, r_loss, color=C_ACT, ls="--", lw=2.0, zorder=3)
ax.set_xlim(1.0, 0.0)
ax.set_xlabel("Aggregate exceedance probability")
ax.set_ylabel("Annual household loss (% of median household income)", fontsize=15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

o_gap, r_gap = o_gul[-1] - o_loss[-1], r_gul[-1] - r_loss[-1]
xo = ep[-1] + 0.012
ax.annotate("", xy=(xo, o_gul[-1]), xytext=(xo, o_loss[-1]),
            arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.6))
ax.annotate(f"Homeowner gap ≈ {o_gap:.0f}%",
            xy=(xo, (o_gul[-1] + o_loss[-1]) / 2), xytext=(0.46, o_gul[-1] * 0.80),
            fontsize=12.5, fontweight="bold", color="0.2", ha="center", va="center", zorder=15,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", lw=0.6, alpha=1.0),
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6, zorder=14))
ax.annotate("", xy=(xo, r_gul[-1]), xytext=(xo, r_loss[-1]),
            arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.4))
ax.annotate(f"Renter gap ≈ {r_gap:.0f}%",
            xy=(xo, (r_gul[-1] + r_loss[-1]) / 2), xytext=(0.46, o_gul[-1] * 0.52),
            fontsize=12, fontweight="bold", color="0.2", ha="center", va="center", zorder=15,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", lw=0.6, alpha=1.0),
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6, zorder=14))

# Legend matches Fig 6 e/f: GUL (red) + Actual loss (blue) only. Tenure (solid =
# Homeowner, dashed = Renter) is conveyed by the gap annotations and curve order.
color_handles = [Line2D([0], [0], color=C_GUL, lw=2.6, label="GUL"),
                 Line2D([0], [0], color=C_ACT, lw=2.6, label="Actual loss")]
ax.legend(handles=color_handles, loc="upper left", fontsize=13,
          framealpha=0.92, edgecolor="#cccccc")
fig.tight_layout()
for p in [OUT / "aep_income_ratio.png", FIG_DIR / "Fig11_income_normalized.png"]:
    fig.savefig(p, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n=== income-normalized AEP (one axis, no title) ===")
print(f"  owner : GUL {o_gul[-1]:5.1f}%  actual {o_loss[-1]:5.1f}%  gap {o_gap:4.1f}% of income")
print(f"  renter: GUL {r_gul[-1]:5.1f}%  actual {r_loss[-1]:5.1f}%  gap {r_gap:4.1f}% of income")
print(f"saved -> {OUT/'aep_income_ratio.png'}  AND  {FIG_DIR/'Fig11_income_normalized.png'}")
