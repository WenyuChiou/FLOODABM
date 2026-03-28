# core/visualizer.py
# Plotting and Reporting — Owner/Renter TP Decay

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from core.model import precompute_eff_grid, LN2, eff_closed_form

# --- Design Tokens ---
STYLE_CONF = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
}

COLORS = {
    "owner": "#2171B5",       # Blue (Homeowner)
    "owner_fill": "#6BAED6",  # Light Blue
    "renter": "#238B45",      # Green (Renter) — matches main text
    "renter_fill": "#74C476", # Light Green
}

def set_pub_style():
    plt.rcParams.update(STYLE_CONF)

def save_results(best_pair: dict, owner_grid: pd.DataFrame, renter_grid: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    owner_grid.to_excel(outdir / "owner_grid_search_full.xlsx", index=False)
    renter_grid.to_excel(outdir / "renter_grid_search_full.xlsx", index=False)

    summary_data = [
        {"Group": "owner",  **best_pair["owner"].to_dict()},
        {"Group": "renter", **best_pair["renter"].to_dict()},
        {"Metric": "Pair_RMSE_Sum", "Value": best_pair["score"]},
        {"Metric": "Alpha_Gap", "Value": best_pair["gap_alpha"]},
        {"Metric": "Beta_Gap",  "Value": best_pair["gap_beta"]}
    ]
    pd.DataFrame(summary_data).to_excel(outdir / "calibration_summary.xlsx", index=False)
    print(f"Results saved to {outdir}")

def plot_overlay(best_pair: dict, owner_raw_df: pd.DataFrame, renter_raw_df: pd.DataFrame, outdir: Path):
    set_pub_style()
    t = np.linspace(0, cfg.COMPUTE["t_max"], 801)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for tag, raw_df, color_key in [("owner", owner_raw_df, "owner"), ("renter", renter_raw_df, "renter")]:
        row = best_pair[tag]
        a, b = row["alpha"], row["beta"]
        t0, t_inf, k = row["tau0"], row["tau_inf"], row["k"]

        PA_bar, SC_bar = raw_df["PA"].mean(), raw_df["SC"].mean()

        tg, eg = precompute_eff_grid(cfg.COMPUTE["t_max"], cfg.COMPUTE["dt"], t0, t_inf, k)
        Eff = np.interp(t, tg, eg)
        w_eff = a * PA_bar + b * SC_bar
        TP = np.exp(-LN2 * w_eff * Eff)

        label = f"{tag.capitalize()} (RMSE={row['RMSE']:.3f})"
        ax.plot(t, TP, label=label, color=COLORS[color_key])

    ax.axvline(5, linestyle=":", color="gray", alpha=0.8, linewidth=1.5)
    ax.text(5.2, 0.95, "Flood (Year 5)", fontsize=8, color="dimgray", va="top")

    ax.set_title("Calibrated Threat Perception Decay (Owner vs Renter)")
    ax.set_xlabel("Years since Flood")
    ax.set_ylabel("Relative Threat Perception TP(t)")
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")

    fig.tight_layout()
    fig.savefig(outdir / "tp_decay_curves.png")
    fig.savefig(outdir / "tp_decay_curves.pdf")
    plt.close(fig)
    print("[OK] tp_decay_curves")

def plot_confidence_intervals(best_pair, owner_boot, renter_boot, owner_raw, renter_raw, outdir):
    set_pub_style()
    t = np.linspace(0, cfg.COMPUTE["t_max"], 801)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    def get_ribbon(boot_df, row, raw_df):
        a, b = row["alpha"], row["beta"]
        PA_bar, SC_bar = raw_df["PA"].mean(), raw_df["SC"].mean()
        w_eff = a * PA_bar + b * SC_bar

        curves = []
        for _, sample in boot_df.iterrows():
            Eff = eff_closed_form(t, sample["tau0"], sample["tau_inf"], sample["k"])
            curves.append(np.exp(-LN2 * w_eff * Eff))
        curves = np.array(curves)

        lo = np.percentile(curves, 2.5, axis=0)
        hi = np.percentile(curves, 97.5, axis=0)
        med = np.percentile(curves, 50.0, axis=0)
        return lo, med, hi

    lo, med, hi = get_ribbon(owner_boot, best_pair["owner"], owner_raw)
    ax.fill_between(t, lo, hi, color=COLORS["owner_fill"], alpha=0.5, label="Homeowner 95% CI")
    ax.plot(t, med, color=COLORS["owner"], linestyle="-", label="Homeowner Median")

    lo, med, hi = get_ribbon(renter_boot, best_pair["renter"], renter_raw)
    ax.fill_between(t, lo, hi, color=COLORS["renter_fill"], alpha=0.5, label="Renter 95% CI")
    ax.plot(t, med, color=COLORS["renter"], linestyle="-", label="Renter Median")

    ax.set_xlabel("Years since Flood")
    ax.set_ylabel("Relative TP(t)")
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(outdir / "tp_decay_confidence_intervals.png")
    fig.savefig(outdir / "tp_decay_confidence_intervals.pdf")
    plt.close(fig)
    print("[OK] tp_decay_confidence_intervals")

def plot_residuals(best_pair, owner_df, renter_df, outdir):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(5, 5))

    def get_preds(row, df):
        a, b = row["alpha"], row["beta"]
        t0, tinf, k = row["tau0"], row["tau_inf"], row["k"]
        times = 0.5 * (df["t_lo"] + df["t_hi"])
        PA, SC = df["PA"], df["SC"]
        w_eff = a * (1.0 - PA) + b * SC
        Eff = eff_closed_form(times.values, t0, tinf, k)
        g = np.exp(-LN2 * w_eff * Eff)
        return row["TP0_hat"] * g

    owner_pred = get_preds(best_pair["owner"], owner_df)
    renter_pred = get_preds(best_pair["renter"], renter_df)

    ax.scatter(owner_df["TP"], owner_pred, color=COLORS["owner"], alpha=0.6, s=25,
               label=f"Owner (n={len(owner_df)})", edgecolors='w', linewidth=0.5)
    ax.scatter(renter_df["TP"], renter_pred, color=COLORS["renter"], alpha=0.6, s=25,
               label=f"Renter (n={len(renter_df)})", edgecolors='w', linewidth=0.5)

    lims = [0, 1]
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, linewidth=1)

    ax.set_xlabel("Observed TP")
    ax.set_ylabel("Predicted TP")
    ax.set_title("Goodness of Fit: Observed vs Predicted TP")
    ax.legend(loc="upper left")

    rmse_owner = best_pair["owner"]["RMSE"]
    rmse_renter = best_pair["renter"]["RMSE"]
    stats = f"RMSE (Owner): {rmse_owner:.3f}\nRMSE (Renter): {rmse_renter:.3f}"
    ax.text(0.95, 0.05, stats, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(outdir / "tp_decay_residuals.png")
    fig.savefig(outdir / "tp_decay_residuals.pdf")
    plt.close(fig)
    print("[OK] tp_decay_residuals")

def plot_rmse_heatmap(grid_df: pd.DataFrame, group_name: str, outdir: Path):
    """RMSE landscape heatmap for grid search results."""
    set_pub_style()
    pivot = grid_df.pivot_table(index="beta", columns="alpha", values="RMSE")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values, cmap="viridis_r", shading="auto")
    fig.colorbar(im, ax=ax, label="RMSE")
    ax.set_xlabel(r"$\alpha$ (Place Attachment weight)")
    ax.set_ylabel(r"$\beta$ (Social Capital weight)")
    ax.set_title(f"RMSE Landscape — {group_name.capitalize()}")
    fig.tight_layout()
    fig.savefig(outdir / f"rmse_heatmap_{group_name}.png")
    fig.savefig(outdir / f"rmse_heatmap_{group_name}.pdf")
    plt.close(fig)
    print(f"[OK] rmse_heatmap_{group_name}")
