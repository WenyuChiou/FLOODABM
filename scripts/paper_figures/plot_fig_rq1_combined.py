# -*- coding: utf-8 -*-
"""
Fig RQ1 Combined — 4-row layout for merged RQ1 (damage + finance)
  Row 1 (a,b): Cumulative damage + actual loss per HH — Owner | Renter  [old Fig4]
  Row 2 (c):   Damage/Loss ratio Owner vs Renter (single, full width)
  Row 3 (d,e): Premium + OOP stacked bar + OOP rate — Owner | Renter    [old Fig8]
  Row 4 (f,g): Payout bar + Payout rate — Owner | Renter               [old Fig8]
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pathlib import Path
import glob, re

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent  # project root
sys.path.insert(0, str(ROOT))
from utils.plots_modular.style import set_paper_style, COLORS
set_paper_style()

# ── Paths ──
# 50-run Monte Carlo outputs live on the local disk (not OneDrive) to avoid
# sync pressure. Each run writes to
#   C:\FLOODABM_mc50\{scenario}\run_XX\baseline\{scenario}
# where scenario ∈ {baseline, worst}.
MC_ROOT = Path(r"C:\FLOODABM_mc50_v2")
N_RUNS = 50

def run_path(scenario: str, run_id: int) -> Path:
    return MC_ROOT / scenario / f"run_{run_id:02d}" / "baseline" / scenario

# Legacy single-run dirs — kept only for visualization output paths.
BASE = ROOT / "outputs" / "baseline" / "baseline"
WORST = ROOT / "outputs" / "baseline" / "worst"
VIS = BASE / "visualization" / "compare"
VIS.mkdir(parents=True, exist_ok=True)
SM_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\draft\v2_sections\SM")
SM_DIR.mkdir(parents=True, exist_ok=True)

SEVERE = [2011, 2021]

# ── Style ──
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 14, "axes.titlesize": 19, "axes.titleweight": "bold",
    "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 13, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.12, "grid.linestyle": "--",
    "axes.spines.top": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})

# ── Color palette (consistent semantic meaning throughout all panels) ──
# Owner = blue family, Renter = green family
# Within Row 1: solid = Baseline (adaptation), dashed-lighter = No-adaptation (worst)
C_OWN_BASE  = "#1a6faf"   # Owner  — Baseline     (solid blue)
C_OWN_WORST = "#6baed6"   # Owner  — No-adapt     (lighter blue)
C_REN_BASE  = "#1f7a3c"   # Renter — Baseline     (solid green)
C_REN_WORST = "#74c476"   # Renter — No-adapt     (lighter green)

# Finance panels: stacked bars use two tones within each group's hue
C_PREM_O  = "#1a6faf"     # Owner  premium  (blue)
C_OOP_O   = "#9ecae1"     # Owner  OOP      (pale blue)
C_PAY_O   = "#08519c"     # Owner  payout   (dark blue)
C_PREM_R  = "#1f7a3c"     # Renter premium  (green)
C_OOP_R   = "#a1d99b"     # Renter OOP      (pale green)
C_PAY_R   = "#006d2c"     # Renter payout   (dark green)

# Ratio panel uses owner/renter colors
C_OWNER  = C_OWN_BASE
C_RENTER = C_REN_BASE

# Severe year shading
C_SEVERE = "#e0e0e0"

def usd_fmt(x, _):
    ax = abs(x)
    if ax >= 1e6: return f"${x/1e6:.1f}M"
    if ax >= 1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"

def shade_severe(ax, years_list):
    for i, y in enumerate(years_list):
        if y in SEVERE:
            ax.axvspan(i - 0.45, i + 0.45, color=C_SEVERE, alpha=0.55, zorder=0,
                       label="_nolegend_")

def plabel(ax, text, fontsize=18):
    """Bold panel label in upper-left, outside axes."""
    ax.text(-0.10, 1.04, text, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", ha="left", va="bottom")

def set_xticks(ax, x, years, step=2):
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(y) for y in years[::step]], rotation=0)
    ax.set_xlim(x[0] - 0.7, x[-1] + 1.2)   # right margin for annotations

def twin_ax_style(ax2, ylabel, ylim=None, n_ticks=5):
    """Consistent secondary y-axis formatting."""
    ax2.set_ylabel(ylabel, fontsize=18, color="#444444")
    ax2.tick_params(axis="y", labelsize=15, colors="#444444")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#bbbbbb")
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks, prune="both"))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    if ylim is not None:
        ax2.set_ylim(0, ylim)

# ══════════════════════════════════════════════════════════
# DATA LOADING — 50-run Monte Carlo (tract-level)
# ══════════════════════════════════════════════════════════
# All finance quantities are read from tract-level CSVs (finance_tract_*.csv),
# which are preserved by --output-mode summary. Per-tenure OOP and payout are
# split proportionally by owner/renter gross damage — matching the approach
# already used in the original Fig4 ratio computation. Premium uses the exact
# owner_premium_total_usd / renter_premium_total_usd columns from the tract CSV.
def load_scenario_tract(root):
    fin_frames = []
    for f in sorted(glob.glob(str(root / "finance/finance_tract_*.csv"))):
        m = re.search(r"tract_(\d{4})\.csv", f)
        if m:
            df = pd.read_csv(f, dtype={"tract_geoid": str})
            df["year"] = int(m.group(1))
            fin_frames.append(df)
    fin = pd.concat(fin_frames, ignore_index=True)
    # Per-tenure payout/OOP columns come from the runner.py aggregation patch.
    # Rate denominators use `owner_gross_total_usd` / `renter_gross_total_usd`
    # (= vulnerability-module gross damage summed over ALL households, insured
    # or not) to match the original Fig5 interpretation: "OOP as a % of total
    # gross damage", not "OOP as a % of only the insured portion".
    fin_yr = fin.groupby("year").agg(
        owner_gross=("owner_gross_total_usd", "sum"),
        renter_gross=("renter_gross_total_usd", "sum"),
        payout_total=("payout_total_usd", "sum"),
        oop_total=("oop_total_usd", "sum"),
        owner_payout=("owner_payout_total_usd", "sum"),
        renter_payout=("renter_payout_total_usd", "sum"),
        owner_oop=("owner_oop_total_usd", "sum"),
        renter_oop=("renter_oop_total_usd", "sum"),
        owner_premium=("owner_premium_total_usd", "sum"),
        renter_premium=("renter_premium_total_usd", "sum"),
        owner_hh=("owner_households", "sum"),
        renter_hh=("renter_households", "sum"),
    ).reset_index().fillna(0).sort_values("year").reset_index(drop=True)

    # Per-tenure gross damage used by Fig4 cumulative damage/loss.
    fin_yr["owner_dmg"] = fin_yr["owner_gross"]
    fin_yr["renter_dmg"] = fin_yr["renter_gross"]

    # Per-HH averages (ALL households, insured or not) for Fig5 bars.
    o_hh = np.maximum(fin_yr["owner_hh"].values, 1)
    r_hh = np.maximum(fin_yr["renter_hh"].values, 1)
    fin_yr["owner_premium_ph"] = fin_yr["owner_premium"] / o_hh
    fin_yr["renter_premium_ph"] = fin_yr["renter_premium"] / r_hh
    fin_yr["owner_oop_ph"] = fin_yr["owner_oop"] / o_hh
    fin_yr["renter_oop_ph"] = fin_yr["renter_oop"] / r_hh
    fin_yr["owner_payout_ph"] = fin_yr["owner_payout"] / o_hh
    fin_yr["renter_payout_ph"] = fin_yr["renter_payout"] / r_hh

    # OOP / payout rates (% of tenure-specific gross damage) for Fig5 dashed lines.
    ow_g = fin_yr["owner_gross"].values
    re_g = fin_yr["renter_gross"].values
    fin_yr["owner_oop_rate"] = np.where(ow_g > 0, fin_yr["owner_oop"].values / ow_g * 100, 0)
    fin_yr["renter_oop_rate"] = np.where(re_g > 0, fin_yr["renter_oop"].values / re_g * 100, 0)
    fin_yr["owner_payout_rate"] = np.where(ow_g > 0, fin_yr["owner_payout"].values / ow_g * 100, 0)
    fin_yr["renter_payout_rate"] = np.where(re_g > 0, fin_yr["renter_payout"].values / re_g * 100, 0)
    return fin_yr

def load_all_runs(scenario):
    """Stack load_scenario_tract outputs across all 50 MC runs.
    Returns a dict keyed by column name. Scalar columns become (n_runs, n_years)
    arrays; 'year' stays a 1-D array."""
    frames = []
    for run_id in range(1, N_RUNS + 1):
        d = run_path(scenario, run_id)
        if not d.exists():
            print(f"  [warn] missing {d}")
            continue
        frames.append(load_scenario_tract(d))
    if not frames:
        raise FileNotFoundError(f"No MC runs found under {MC_ROOT / scenario}")
    years_arr = frames[0]["year"].values
    out = {"year": years_arr, "_n_runs": len(frames)}
    for col in frames[0].columns:
        if col == "year":
            continue
        out[col] = np.stack([f[col].values.astype(float) for f in frames], axis=0)
    return out

def cum_series_all(data, group):
    """Return (cum_damage, cum_loss) per-HH, shape (n_runs, n_years)."""
    hh = np.maximum(data[f"{group}_hh"], 1)
    dmg = data[f"{group}_dmg"]
    pay = data[f"{group}_payout"]
    cum_dmg = np.cumsum(dmg / hh, axis=1)
    cum_loss = np.cumsum((dmg - pay) / hh, axis=1)
    return cum_dmg, cum_loss

def cum_ratio_per_run(base_d, worst_d, group):
    """Per-run ratio = Δ cum damage / Δ cum actual loss, (n_runs, n_years)."""
    cdB, clB = cum_series_all(base_d, group)
    cdW, clW = cum_series_all(worst_d, group)
    delta_dmg = cdW - cdB
    delta_loss = clW - clB
    return np.where(delta_loss > 0, delta_dmg / delta_loss, 1.0)

def stats(arr):
    """Axis-0 median / q25 / q75 for an (n_runs, n_years) array."""
    return (np.median(arr, axis=0),
            np.percentile(arr, 25, axis=0),
            np.percentile(arr, 75, axis=0))

# ── Load all 50 runs for both scenarios ──
print("Loading 50-run Monte Carlo outputs...")
base_d = load_all_runs("baseline")
worst_d = load_all_runs("worst")
print(f"  baseline: {base_d['_n_runs']} runs loaded")
print(f"  worst:    {worst_d['_n_runs']} runs loaded")

years = list(base_d["year"])
x = np.arange(len(years))

cdB_O_all, clB_O_all = cum_series_all(base_d, "owner")
cdW_O_all, clW_O_all = cum_series_all(worst_d, "owner")
cdB_R_all, clB_R_all = cum_series_all(base_d, "renter")
cdW_R_all, clW_R_all = cum_series_all(worst_d, "renter")
ratio_O_all = cum_ratio_per_run(base_d, worst_d, "owner")
ratio_R_all = cum_ratio_per_run(base_d, worst_d, "renter")

cdB_O, cdB_O_q25, cdB_O_q75 = stats(cdB_O_all)
clB_O, clB_O_q25, clB_O_q75 = stats(clB_O_all)
cdW_O, cdW_O_q25, cdW_O_q75 = stats(cdW_O_all)
clW_O, clW_O_q25, clW_O_q75 = stats(clW_O_all)
cdB_R, cdB_R_q25, cdB_R_q75 = stats(cdB_R_all)
clB_R, clB_R_q25, clB_R_q75 = stats(clB_R_all)
cdW_R, cdW_R_q25, cdW_R_q75 = stats(cdW_R_all)
clW_R, clW_R_q25, clW_R_q75 = stats(clW_R_all)
ratio_O, ratio_O_q25, ratio_O_q75 = stats(ratio_O_all)
ratio_R, ratio_R_q25, ratio_R_q75 = stats(ratio_R_all)

print("=== Cumulative per HH, 2023 endpoint (median [Q25–Q75]) ===")
print(f"  Owner  damage: ${cdB_O[-1]:,.0f} [{cdB_O_q25[-1]:,.0f}-{cdB_O_q75[-1]:,.0f}] (base)"
      f"  ${cdW_O[-1]:,.0f} [{cdW_O_q25[-1]:,.0f}-{cdW_O_q75[-1]:,.0f}] (no-adapt)")
print(f"  Owner  loss:   ${clB_O[-1]:,.0f} [{clB_O_q25[-1]:,.0f}-{clB_O_q75[-1]:,.0f}] (base)"
      f"  ${clW_O[-1]:,.0f} [{clW_O_q25[-1]:,.0f}-{clW_O_q75[-1]:,.0f}] (no-adapt)")
print(f"  Renter damage: ${cdB_R[-1]:,.0f} [{cdB_R_q25[-1]:,.0f}-{cdB_R_q75[-1]:,.0f}] (base)"
      f"  ${cdW_R[-1]:,.0f} [{cdW_R_q25[-1]:,.0f}-{cdW_R_q75[-1]:,.0f}] (no-adapt)")
print(f"  Renter loss:   ${clB_R[-1]:,.0f} [{clB_R_q25[-1]:,.0f}-{clB_R_q75[-1]:,.0f}] (base)"
      f"  ${clW_R[-1]:,.0f} [{clW_R_q25[-1]:,.0f}-{clW_R_q75[-1]:,.0f}] (no-adapt)")
print(f"  Owner  ratio 2023: {ratio_O[-1]:.3f} [{ratio_O_q25[-1]:.3f}-{ratio_O_q75[-1]:.3f}]")
print(f"  Renter ratio 2023: {ratio_R[-1]:.3f} [{ratio_R_q25[-1]:.3f}-{ratio_R_q75[-1]:.3f}]")

# ══════════════════════════════════════════════════════════
# FIGURE 4 — Cumulative damage/loss (2 panels) + Ratio (1 panel)
#             Layout: 1×3 (owner | renter | ratio)
# ══════════════════════════════════════════════════════════
import matplotlib.gridspec as gridspec

# --- Colors for cross-scenario distinction (CLEAR contrast) ---
C_DMG  = "#d62828"   # GUL = red (both scenarios)
C_LOSS = "#0077b6"   # Actual loss = blue (both scenarios)
# Scenario distinction: solid+filled = Baseline, dashed+open = No-adaptation

# Per-figure rc overrides: paper Fig5 keeps the original (smaller) fonts;
# Fig6 (built later) re-applies the enlarged rcParams set above.
_FIG5_RC = {
    "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
}
_FIG6_RC = {
    "font.size": 14, "axes.titlesize": 19,
    "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 13,
}
plt.rcParams.update(_FIG5_RC)

fig4 = plt.figure(figsize=(10.5, 4.8))
gs4 = fig4.add_gridspec(1, 2, wspace=0.28, width_ratios=[1, 1])

for col, (cdB, cdW, clB, clW, cdB_lo, cdB_hi, cdW_lo, cdW_hi,
          clB_lo, clB_hi, clW_lo, clW_hi, title, plbl) in enumerate([
    (cdB_O, cdW_O, clB_O, clW_O,
     cdB_O_q25, cdB_O_q75, cdW_O_q25, cdW_O_q75,
     clB_O_q25, clB_O_q75, clW_O_q25, clW_O_q75, "Homeowner", "(a)"),
    (cdB_R, cdW_R, clB_R, clW_R,
     cdB_R_q25, cdB_R_q75, cdW_R_q25, cdW_R_q75,
     clB_R_q25, clB_R_q75, clW_R_q25, clW_R_q75, "Renter", "(b)"),
]):
    ax = fig4.add_subplot(gs4[0, col])
    shade_severe(ax, years)

    # Per-year IQR error bars (Q25–Q75) on the two baseline (solid) lines.
    # Use BLACK error bars drawn ABOVE the data markers (high zorder) so the
    # caps contrast against the colored lines and remain visible even when
    # the renter loss variance is small. The two no-adaptation (dashed) lines
    # are deterministic counterfactuals and get no error bars.
    def _err_pair(med, lo, hi):
        return np.vstack([med - lo, hi - med])
    ax.errorbar(x, cdB, yerr=_err_pair(cdB, cdB_lo, cdB_hi),
                fmt="none", ecolor="black", elinewidth=1.2,
                capsize=5.0, capthick=1.5, zorder=8)
    ax.errorbar(x, clB, yerr=_err_pair(clB, clB_lo, clB_hi),
                fmt="none", ecolor="black", elinewidth=1.2,
                capsize=5.0, capthick=1.5, zorder=8)

    # Baseline (solid, filled markers)
    l1, = ax.plot(x, cdB, "-o",  ms=5, lw=2.2, color=C_DMG,
                  label="Adaptation — GUL", zorder=4)
    l2, = ax.plot(x, clB, "-s",  ms=5, lw=2.2, color=C_LOSS,
                  label="Adaptation — Actual loss", zorder=4)
    # No-adaptation (dashed, open markers)
    l3, = ax.plot(x, cdW, "o",  ms=5, lw=1.8, color=C_DMG,
                  linestyle="--", markerfacecolor="white", markeredgewidth=1.5,
                  label="No-adapt baseline — GUL", zorder=3)
    l4, = ax.plot(x, clW, "s",  ms=5, lw=1.8, color=C_LOSS,
                  linestyle="--", markerfacecolor="white", markeredgewidth=1.5,
                  label="No-adapt baseline — Actual loss", zorder=3)

    # Shaded gaps between scenarios
    ax.fill_between(x, cdB, cdW, alpha=0.08, color=C_DMG, zorder=1)
    ax.fill_between(x, clB, clW, alpha=0.08, color=C_LOSS, zorder=1)

    ax.set_ylabel("Cumulative per HH ($)", fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    set_xticks(ax, x, years)
    ax.set_xlabel("Year", fontsize=11)
    ax.spines["right"].set_visible(False)
    plabel(ax, plbl, fontsize=11)
    ax.legend(handles=[l1, l3, l2, l4], loc="upper left", fontsize=9,
              framealpha=0.95, edgecolor="#cccccc", handlelength=2.5,
              labelspacing=0.3, borderpad=0.4, handletextpad=0.4)

# Panel (c) Insurance Leverage Ratio removed — the ratio is a derived quantity
# (Δ damage / Δ actual loss) whose denominator becomes small for homeowners,
# amplifying posterior-sampling uncertainty to a visually misleading degree.
# Leverage-ratio rows remain in Table_S_stochastic_variance.csv for reference.

# ── Endpoint annotations: 50-run median $ value only (no brackets) ──
for col, (cdB, cdW, clB, clW) in enumerate([
    (cdB_O, cdW_O, clB_O, clW_O),
    (cdB_R, cdW_R, clB_R, clW_R),
]):
    ax = fig4.axes[col]
    x_end = x[-1]

    # Dollar labels at 2023 endpoint — 50-run median only
    labels = [
        (cdW[-1], f"${cdW[-1]/1e3:.0f}K", C_DMG),
        (cdB[-1], f"${cdB[-1]/1e3:.0f}K", C_DMG),
        (clW[-1], f"${clW[-1]/1e3:.0f}K", C_LOSS),
        (clB[-1], f"${clB[-1]/1e3:.0f}K", C_LOSS),
    ]
    labels.sort(key=lambda t: t[0])
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    min_gap = y_range * 0.05
    ys = [l[0] for l in labels]
    for i in range(1, len(ys)):
        if ys[i] - ys[i-1] < min_gap:
            ys[i] = ys[i-1] + min_gap
    for i in range(len(ys)-2, -1, -1):
        if ys[i+1] - ys[i] < min_gap:
            ys[i] = ys[i+1] - min_gap

    for (_, txt, color), y_pos in zip(labels, ys):
        ax.text(x_end + 0.3, y_pos, txt,
                fontsize=10, fontweight="bold", color=color,
                ha="left", va="center", clip_on=False,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85))

fig4.tight_layout()
_PAPER_FIG = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
                  r"\NSF-project\ABM\paper\Figure")
for path in [VIS / "fig4_damage_ratio.png", VIS / "fig4_damage_ratio.pdf",
             SM_DIR / "fig4_damage_ratio.png", SM_DIR / "fig4_damage_ratio.pdf",
             _PAPER_FIG / "Fig5_damage_ratio.png", _PAPER_FIG / "Fig5_damage_ratio.pdf"]:
    fig4.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig4)
print(f"\nSaved Fig 4: {VIS / 'fig4_damage_ratio.png'}")

# Restore enlarged rcParams for paper Fig6 (the finance multi-panel below).
plt.rcParams.update(_FIG6_RC)

# Fig5 data — reuse the 50-run tract-level aggregates in base_d.
# Bars show 50-run median per HH with per-year IQR error bars on top;
# rate lines carry a 50-run IQR band.
o_premium = np.median(base_d["owner_premium_ph"], axis=0)
o_oop = np.median(base_d["owner_oop_ph"], axis=0)
o_payout = np.median(base_d["owner_payout_ph"], axis=0)
o_oop_rate, o_oop_rate_q25, o_oop_rate_q75 = stats(base_d["owner_oop_rate"])
o_payout_rate, o_payout_rate_q25, o_payout_rate_q75 = stats(base_d["owner_payout_rate"])

r_premium = np.median(base_d["renter_premium_ph"], axis=0)
r_oop = np.median(base_d["renter_oop_ph"], axis=0)
r_payout = np.median(base_d["renter_payout_ph"], axis=0)
r_oop_rate, r_oop_rate_q25, r_oop_rate_q75 = stats(base_d["renter_oop_rate"])
r_payout_rate, r_payout_rate_q25, r_payout_rate_q75 = stats(base_d["renter_payout_rate"])

# Per-year IQR for stacked bar totals (Premium + OOP) and Payout bars.
def _total_q(col_a, col_b):
    total_runs = base_d[col_a] + base_d[col_b]
    return stats(total_runs)  # returns median, q25, q75
o_stack_med, o_stack_q25, o_stack_q75 = _total_q("owner_premium_ph", "owner_oop_ph")
r_stack_med, r_stack_q25, r_stack_q75 = _total_q("renter_premium_ph", "renter_oop_ph")
o_pay_med, o_pay_q25, o_pay_q75 = stats(base_d["owner_payout_ph"])
r_pay_med, r_pay_q25, r_pay_q75 = stats(base_d["renter_payout_ph"])

# Color for the IQR band on rate lines — neutral grey, subtle.
C_BAND = "#777777"

# ══════════════════════════════════════════════════════════
# FIGURE 8 — Finance: 3×2 (premium+OOP | payout | rank-prob)
# ══════════════════════════════════════════════════════════
fig8, axes8 = plt.subplots(3, 2, figsize=(12, 12.5))

# Row 3 data: PER-HOUSEHOLD annual ground-up loss and actual loss pooled across
# 50 runs × 12 years = 600 values per tenure (warm-up year 2011 excluded
# because it is deterministic across runs — no decisions have occurred yet,
# so all 50 runs share identical GUL / payout / actual loss in 2011 and
# would otherwise create a flat plateau at the top of the rank plot).
#
# PER-HH normalization (reframe AEP from aggregate to per-household): each
# (run, year) aggregate tenure loss is divided by THAT (run, year) tenure
# household count BEFORE pooling. This matters because owner households decline
# over time (relocation / buyout: 41,517 in 2011 → 37,689 in 2023) while renters
# stay at 10,624 — a single constant divisor would bias the curve. Dividing per
# (run, year) keeps the normalization exact across the stochastic household
# trajectories of the 50 MC runs.
_o_hh = np.maximum(base_d["owner_hh"][:, 1:],  1.0)
_r_hh = np.maximum(base_d["renter_hh"][:, 1:], 1.0)
owner_gul_flat   = (base_d["owner_gross"][:, 1:]                                   / _o_hh).ravel()
owner_loss_flat  = ((base_d["owner_gross"][:, 1:]  - base_d["owner_payout"][:, 1:])  / _o_hh).ravel()
renter_gul_flat  = (base_d["renter_gross"][:, 1:]                                  / _r_hh).ravel()
renter_loss_flat = ((base_d["renter_gross"][:, 1:] - base_d["renter_payout"][:, 1:]) / _r_hh).ravel()

# ── Row 1: Premium + OOP ──
oop_ylim = max(o_oop_rate.max(), r_oop_rate.max()) * 1.35
oop_ylim = max(oop_ylim, 20)

for col, (prem, oop, oop_r, oop_r_lo, oop_r_hi, c_prem, c_oop, stack_med, stack_lo, stack_hi, title, plbl) in enumerate([
    (o_premium, o_oop, o_oop_rate, o_oop_rate_q25, o_oop_rate_q75,
     C_PREM_O, C_OOP_O, o_stack_med, o_stack_q25, o_stack_q75, "Homeowner", "(a)"),
    (r_premium, r_oop, r_oop_rate, r_oop_rate_q25, r_oop_rate_q75,
     C_PREM_R, C_OOP_R, r_stack_med, r_stack_q25, r_stack_q75, "Renter", "(b)"),
]):
    ax = axes8[0, col]
    shade_severe(ax, years)
    ax.bar(x, prem, 0.65, label="Premium",  color=c_prem, alpha=0.85, zorder=3)
    ax.bar(x, oop,  0.65, bottom=prem, label="OOP cost", color=c_oop,  alpha=0.85, zorder=3)
    # Error bar at the top of the stacked bar (Premium + OOP total IQR)
    stack_top = prem + oop  # median total
    ax.errorbar(x, stack_top, yerr=np.vstack([stack_top - stack_lo, stack_hi - stack_top]),
                fmt="none", ecolor="black", elinewidth=1.2, capsize=4.0, capthick=1.2, zorder=6)
    ax.set_ylabel("Avg per HH ($)", fontsize=18)
    ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.set_title(title, fontsize=19, fontweight="bold", pad=8)
    set_xticks(ax, x, years)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    plabel(ax, plbl)
    ax2 = ax.twinx()
    ax2.fill_between(x, oop_r_lo, oop_r_hi, color=C_BAND, alpha=0.25, zorder=4, linewidth=0)
    ax2.plot(x, oop_r, color="#444444", marker="o", ms=4, lw=1.6,
             linestyle="--", label="OOP rate", zorder=5)
    twin_ax_style(ax2, "Share of GUL (%)", ylim=oop_ylim, n_ticks=4)
    # Panel (b) Renter: extend ylim ~35% above the tallest stacked bar so the
    # upper-right legend has clear headroom and does not overlap the 2021 bar.
    if plbl == "(b)":
        y_top = float((prem + oop + (stack_hi - stack_top)).max())
        ax.set_ylim(top=y_top * 1.35)

    h1, l1_ = ax.get_legend_handles_labels()
    h2, l2_ = ax2.get_legend_handles_labels()
    # Panel (a) Homeowner: legend upper-left; panel (b) Renter: legend upper-right
    # (with extra ylim headroom above so 2021 bar remains visible).
    legend_loc = "upper right" if plbl == "(b)" else "upper left"
    ax.legend(h1 + h2, l1_ + l2_, loc=legend_loc, fontsize=13,
              framealpha=0.92, edgecolor="#cccccc",
              handlelength=1.8, labelspacing=0.3, borderpad=0.4)

# ── Row 2: Payout ──
for col, (pay, pay_r, pay_r_lo, pay_r_hi, c_pay, pay_lo, pay_hi, title, plbl) in enumerate([
    (o_payout, o_payout_rate, o_payout_rate_q25, o_payout_rate_q75,
     C_PAY_O, o_pay_q25, o_pay_q75, "Homeowner", "(c)"),
    (r_payout, r_payout_rate, r_payout_rate_q25, r_payout_rate_q75,
     C_PAY_R, r_pay_q25, r_pay_q75, "Renter", "(d)"),
]):
    ax = axes8[1, col]
    shade_severe(ax, years)
    ax.bar(x, pay, 0.65, label="Payout", color=c_pay, alpha=0.85, zorder=3)
    # Error bar at top of Payout bar (per-year IQR)
    ax.errorbar(x, pay, yerr=np.vstack([pay - pay_lo, pay_hi - pay]),
                fmt="none", ecolor="black", elinewidth=1.2, capsize=4.0, capthick=1.2, zorder=6)
    ax.set_ylabel("Avg payout per HH ($)", fontsize=18)
    ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.set_xlabel("Year", fontsize=18)
    ax.set_title(title, fontsize=19, fontweight="bold", pad=8)
    set_xticks(ax, x, years)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    plabel(ax, plbl)
    ax2 = ax.twinx()
    ax2.fill_between(x, pay_r_lo, pay_r_hi, color=C_BAND, alpha=0.25, zorder=4, linewidth=0)
    ax2.plot(x, pay_r, color="#444444", marker="s", ms=4, lw=1.6,
             linestyle="--", label="Payout rate", zorder=5)
    twin_ax_style(ax2, "Share of GUL (%)", ylim=102, n_ticks=5)
    h1, l1_ = ax.get_legend_handles_labels()
    h2, l2_ = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1_ + l2_, loc="upper left", fontsize=13,
              framealpha=0.92, edgecolor="#cccccc",
              handlelength=1.8, labelspacing=0.3, borderpad=0.4)

# ── Row 3: Exceedance Probability (EP) curves of PER-HH annual loss ──
# Pool 50 runs × 12 years (2012-2023) = 600 per-household annual values.
# x = P(L > x), linear scale, left=1 (frequent) → right=0 (rare).
for col, (gul_flat, loss_flat, title, plbl) in enumerate([
    (owner_gul_flat,  owner_loss_flat,  "Homeowner", "(e)"),
    (renter_gul_flat, renter_loss_flat, "Renter",    "(f)"),
]):
    ax = axes8[2, col]
    n = len(gul_flat)
    sorted_gul  = np.sort(gul_flat)
    sorted_loss = np.sort(loss_flat)
    ep = 1.0 - np.arange(1, n + 1) / n   # ~1 → ~0

    # Per-HH annual loss — SAME unit ($K) for both panels so the owner vs renter
    # magnitudes are directly comparable on the same yardstick (the whole point
    # of the per-household reframe; aggregate $B-vs-$M hid the per-HH gap).
    is_owner = (col == 0)
    scale_div = 1e3              # per-HH dollars -> thousands
    unit_label = "$K"
    ylabel = "Annual loss per HH ($K)"
    if is_owner:
        endpoint_fmt = "${:.0f}K"
        gap_fmt = "Tail gap = ${:.0f}K"
    else:
        endpoint_fmt = "${:.1f}K"
        gap_fmt = "Tail gap = ${:.1f}K"

    ax.plot(ep, sorted_gul  / scale_div, color=C_DMG, lw=2.0,
            label="GUL")
    ax.plot(ep, sorted_loss / scale_div, color=C_LOSS, lw=2.0,
            label="Actual loss")
    ax.set_xlabel("Aggregate exceedance probability", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=19, fontweight="bold", pad=8)
    ax.set_xlim(1.0, 0.0)
    ax.grid(alpha=0.2, linestyle="--")
    plabel(ax, plbl)
    ax.legend(loc="upper left", fontsize=13, framealpha=0.92,
              edgecolor="#cccccc", handlelength=1.8,
              labelspacing=0.3, borderpad=0.4)

    # Tail-end gap annotation (same style as Fig 7, 8)
    ep_tail  = ep[-1]
    gul_v    = sorted_gul[-1]  / scale_div
    loss_v   = sorted_loss[-1] / scale_div
    gap_v    = gul_v - loss_v
    mid_v    = (gul_v + loss_v) / 2
    # Arrow anchored at the rightmost data point (shortened by 5% each end)
    x_anchor = ep_tail + 0.015
    pad      = gap_v * 0.05
    ax.annotate("",
                xy=(x_anchor, gul_v - pad),
                xytext=(x_anchor, loss_v + pad),
                arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.5))
    # Two-line box: Tail gap amount + covered share (gap / GUL = payout / GUL)
    covered_pct = gap_v / gul_v * 100 if gul_v > 0 else 0
    print(f"[per-HH EP] {title:9s} tail(1%): GUL=${gul_v*scale_div:9,.0f}  "
          f"actual=${loss_v*scale_div:9,.0f}  gap=${gap_v*scale_div:9,.0f}  "
          f"covered={covered_pct:.0f}%")
    y_top = gul_v * 1.15
    ax.set_ylim(top=y_top * 1.08)
    ax.annotate(f"{gap_fmt.format(gap_v)}\nCovered = {covered_pct:.0f}%",
                xy=(x_anchor, mid_v),
                xytext=(x_anchor + 0.32, y_top),
                fontsize=14, fontweight="bold", color="0.2",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="0.4", linewidth=0.6, alpha=1.0),
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))

    # Endpoint dollar labels at tail (same style as Fig 4 / Fig 5)
    ax.text(ep_tail - 0.02, gul_v, endpoint_fmt.format(gul_v),
            fontsize=14, fontweight="bold", color=C_DMG,
            ha="right", va="center", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85))
    ax.text(ep_tail - 0.02, loss_v, endpoint_fmt.format(loss_v),
            fontsize=14, fontweight="bold", color=C_LOSS,
            ha="right", va="center", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85))


# Removed: fig8.text (light blue shading note)
fig8.tight_layout()
for path in [VIS / "fig5_finance.png", VIS / "fig5_finance.pdf",
             SM_DIR / "fig5_finance.png", SM_DIR / "fig5_finance.pdf",
             _PAPER_FIG / "Fig6_finance.png", _PAPER_FIG / "Fig6_finance.pdf"]:
    fig8.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig8)
print(f"Saved Fig 5: {VIS / 'fig5_finance.png'}")
print("Done!")
