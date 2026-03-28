# -*- coding: utf-8 -*-
"""
Monte Carlo analysis and visualization.
Reads outputs from run_montecarlo.py and generates:
  (a) Action share trajectories with IQR bands (owner + renter)
  (b) Financial metrics with IQR bands
  (c) Coefficient of variation summary table
"""
import sys, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "--",
    "axes.spines.top": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})

MC_DIR = Path("outputs/montecarlo_v2")
OUT_DIR = MC_DIR / "fig"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEVERE = [2011, 2021]
SEV_COLOR = "#e3f2fd"
YEARS = list(range(2011, 2024))

def usd_fmt(x, _):
    ax = abs(x)
    if ax >= 1e6: return f"${x/1e6:.1f}M"
    if ax >= 1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"

# ── Load all runs ──
df = pd.read_csv(MC_DIR / "mc_metrics_all.csv")
n_runs = len(df)
print(f"Loaded {n_runs} MC runs")

def shade_severe(ax, years):
    for i, y in enumerate(years):
        if y in SEVERE:
            ax.axvspan(i - 0.4, i + 0.4, color=SEV_COLOR, alpha=0.5, zorder=0)

def extract_ts(df, prefix, years):
    """Extract time series: returns (years, median, q25, q75, mean)."""
    vals = []
    for y in years:
        col = f"{prefix}_{y}"
        if col in df.columns:
            vals.append(df[col].values)
        else:
            vals.append(np.full(len(df), np.nan))
    arr = np.array(vals)  # (n_years, n_runs)
    return {
        "median": np.nanmedian(arr, axis=1),
        "q25": np.nanpercentile(arr, 25, axis=1),
        "q75": np.nanpercentile(arr, 75, axis=1),
        "mean": np.nanmean(arr, axis=1),
        "std": np.nanstd(arr, axis=1),
        "min": np.nanmin(arr, axis=1),
        "max": np.nanmax(arr, axis=1),
    }

# ═══════════════════════════════════════════════════════════════
# Figure 1: Action shares with IQR (2×2: owner/renter × FI/other)
# ═══════════════════════════════════════════════════════════════
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
x = np.arange(len(YEARS))

# Check which action columns exist
sample_cols = df.columns.tolist()
# The action share columns come from the per-tract CSV — check format
# They should be like owner_FI_2011, owner_BP_2011, etc.
has_action_shares = any(c.startswith("owner_FI_") for c in sample_cols)

if has_action_shares:
    owner_actions = {"FI": "#1f77b4", "BP": "#d62828", "EH": "#2ca02c", "DN": "#7f7f7f"}
    renter_actions = {"FI": "#1f77b4", "RL": "#ff7f0e", "DN": "#7f7f7f"}

    for col_idx, (actions, group, title) in enumerate([
        (owner_actions, "owner", "Homeowner"),
        (renter_actions, "renter", "Renter"),
    ]):
        ax_share = axs[0, col_idx]
        ax_count = axs[1, col_idx]
        shade_severe(ax_share, YEARS)
        shade_severe(ax_count, YEARS)

        for action, color in actions.items():
            ts = extract_ts(df, f"{group}_{action}", YEARS)
            ax_share.plot(x, ts["median"], color=color, lw=2, label=action)
            ax_share.fill_between(x, ts["q25"], ts["q75"], alpha=0.2, color=color)

        ax_share.set_ylabel("Action share")
        ax_share.set_ylim(0, 1)
        ax_share.set_title(title)
        ax_share.legend(loc="center right", fontsize=7)

        # Household counts
        hh_ts = extract_ts(df, f"{group}_households", YEARS)
        ax_count.plot(x, hh_ts["median"], "k-o", ms=3, lw=1.5)
        ax_count.fill_between(x, hh_ts["q25"], hh_ts["q75"], alpha=0.15, color="gray")
        ax_count.set_ylabel("Household count")
        ax_count.set_xlabel("Year")
        ax_count.set_xticks(x[::2])
        ax_count.set_xticklabels([str(y) for y in YEARS[::2]])

    # Panel labels
    for idx, label in enumerate(["(a)", "(b)", "(c)", "(d)"]):
        r, c = divmod(idx, 2)
        axs[r, c].text(-0.08, 1.03, label, transform=axs[r, c].transAxes,
                       fontsize=12, fontweight="bold", ha="left", va="bottom")

    fig.suptitle(f"Monte Carlo ({n_runs} runs): Action Shares and Household Counts",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUT_DIR / "mc_action_shares_iqr.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")
else:
    plt.close()
    print("[WARN] No action share columns found — skipping action share figure")

# ═══════════════════════════════════════════════════════════════
# Figure 2: Financial metrics with IQR (2×2)
# ═══════════════════════════════════════════════════════════════
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

fin_metrics = [
    ("payout", "Total Payout", "#ff7f0e"),
    ("oop", "Total OOP", "#2ca02c"),
    ("premium", "Total Premium", "#1f77b4"),
    ("total_dmg", "Total Damage", "#d62828"),
]

for idx, (prefix, label, color) in enumerate(fin_metrics):
    r, c = divmod(idx, 2)
    ax = axs[r, c]
    shade_severe(ax, YEARS)

    ts = extract_ts(df, prefix, YEARS)
    ax.plot(x, ts["median"], color=color, lw=2, label=f"Median")
    ax.fill_between(x, ts["q25"], ts["q75"], alpha=0.25, color=color, label="IQR")
    ax.fill_between(x, ts["min"], ts["max"], alpha=0.08, color=color, label="Range")

    ax.set_ylabel("USD")
    ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax.set_title(label)
    ax.legend(loc="best", fontsize=7)

    panel = ["(a)", "(b)", "(c)", "(d)"][idx]
    ax.text(-0.08, 1.03, panel, transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="left", va="bottom")

    if r == 1:
        ax.set_xlabel("Year")
        ax.set_xticks(x[::2])
        ax.set_xticklabels([str(y) for y in YEARS[::2]])

fig.suptitle(f"Monte Carlo ({n_runs} runs): Financial Metrics with Uncertainty Bands",
             fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
p = OUT_DIR / "mc_finance_iqr.png"
fig.savefig(p, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")

# ═══════════════════════════════════════════════════════════════
# Figure 3: Cumulative damage + loss per HH with MC bands
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, group, title, panel in [
    (ax1, "owner", "Homeowner", "(a)"),
    (ax2, "renter", "Renter", "(b)"),
]:
    shade_severe(ax, YEARS)

    # Per run: compute cumulative damage per HH and cumulative loss per HH
    cum_dmg_runs = []
    cum_loss_runs = []
    for _, row in df.iterrows():
        dmg_per_yr = []
        loss_per_yr = []
        for y in YEARS:
            dmg = row.get(f"{group}_dmg_{y}", 0)
            hh = row.get(f"{group}_households_{y}", 1)
            payout = row.get(f"payout_{y}", 0)
            # Split payout proportional to gross
            o_gross = row.get(f"owner_gross_{y}", 1)
            r_gross = row.get(f"renter_gross_{y}", 1)
            total_gross = o_gross + r_gross
            share = o_gross / total_gross if total_gross > 0 else 0.5
            grp_payout = payout * (share if group == "owner" else 1 - share)
            hh = max(hh, 1)
            dmg_per_yr.append(dmg / hh)
            loss_per_yr.append((dmg - grp_payout) / hh)
        cum_dmg_runs.append(np.cumsum(dmg_per_yr))
        cum_loss_runs.append(np.cumsum(loss_per_yr))

    cum_dmg = np.array(cum_dmg_runs)  # (n_runs, n_years)
    cum_loss = np.array(cum_loss_runs)

    # Damage
    med_d = np.median(cum_dmg, axis=0)
    q25_d = np.percentile(cum_dmg, 25, axis=0)
    q75_d = np.percentile(cum_dmg, 75, axis=0)
    ax.plot(x, med_d, "-o", ms=4, lw=2, color="#0077b6", label="Cum. damage (median)")
    ax.fill_between(x, q25_d, q75_d, alpha=0.2, color="#0077b6", label="Damage IQR")

    # Loss
    ax2r = ax.twinx()
    med_l = np.median(cum_loss, axis=0)
    q25_l = np.percentile(cum_loss, 25, axis=0)
    q75_l = np.percentile(cum_loss, 75, axis=0)
    ax2r.plot(x, med_l, "--^", ms=4, lw=2, color="#2e86de", label="Cum. loss (median)")
    ax2r.fill_between(x, q25_l, q75_l, alpha=0.15, color="#2e86de", label="Loss IQR")
    ax2r.spines["top"].set_visible(False)

    ax.set_ylabel("Cum. damage per HH ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax2r.set_ylabel("Cum. actual loss per HH ($)")
    ax2r.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax.set_xlabel("Year")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(y) for y in YEARS[::2]])
    ax.text(-0.08, 1.03, f"{panel} {title}", transform=ax.transAxes,
            fontsize=13, fontweight="bold", ha="left", va="bottom")

    if group == "owner":
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2r.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

fig.suptitle(f"Monte Carlo ({n_runs} runs): Cumulative Damage and Loss per Household",
             fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
p = OUT_DIR / "mc_cumulative_damage_loss_iqr.png"
fig.savefig(p, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")

# ═══════════════════════════════════════════════════════════════
# Summary: Coefficient of Variation table
# ═══════════════════════════════════════════════════════════════
cv_rows = []
for prefix, label in [("payout", "Payout"), ("oop", "OOP"), ("premium", "Premium"),
                       ("total_dmg", "Total Damage"), ("owner_dmg", "Owner Damage"),
                       ("renter_dmg", "Renter Damage"), ("owner_households", "Owner HH Count"),
                       ("renter_households", "Renter HH Count")]:
    for y in YEARS:
        col = f"{prefix}_{y}"
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            cv = std / mean * 100 if mean > 0 else 0
            cv_rows.append({"metric": label, "year": y, "mean": mean, "std": std, "cv_pct": cv})

cv_df = pd.DataFrame(cv_rows)
cv_summary = cv_df.groupby("metric").agg(
    mean_cv=("cv_pct", "mean"),
    max_cv=("cv_pct", "max"),
    min_cv=("cv_pct", "min"),
).reset_index().sort_values("mean_cv", ascending=False)

print("\n" + "=" * 60)
print(f"Coefficient of Variation Summary ({n_runs} MC runs)")
print("=" * 60)
print(cv_summary.to_string(index=False, float_format="%.2f"))

cv_df.to_csv(MC_DIR / "mc_cv_by_year.csv", index=False, encoding="utf-8-sig")
cv_summary.to_csv(MC_DIR / "mc_cv_summary.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved: {MC_DIR / 'mc_cv_summary.csv'}")
print(f"Saved: {MC_DIR / 'mc_cv_by_year.csv'}")
