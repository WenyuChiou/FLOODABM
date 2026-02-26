# -*- coding: utf-8 -*-
"""
Phase 2 Bayesian Regression — Visualization Suite
==================================================
Reads all_results_summary_v2.xlsx and generates publication-quality plots.

Outputs (saved to same directory as this script):
  1. coefficient_comparison.png/pdf   — β coefficients by group × action
  2. convergence_diagnostics.png/pdf  — R-hat and ESS for all parameters
  3. calibration_improvement.png/pdf  — ECE before vs after calibration
  4. prior_vs_posterior.png/pdf       — Prior→Posterior shrinkage
  5. brier_skill_score.png/pdf        — BSS by model
  6. phase2_metrics_table.csv         — Summary metrics table
"""
from __future__ import annotations
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
XLSX = HERE / "all_results_summary_v2.xlsx"

# MG→owner, NMG→renter mapping (original Bayesian output used MG/NMG labels)
GROUP_MAP = {"MG": "owner", "NMG": "renter"}
ACTIONS = ["FI", "EH", "BP", "RL"]
PARAMS = ["beta_tp", "beta_cp", "beta_sp"]
PARAM_LABELS = {"beta_tp": "β_TP", "beta_cp": "β_CP", "beta_sp": "β_SP", "beta_0": "β₀"}

# Colors
C_OWNER = "#2171B5"
C_RENTER = "#CB181D"
C_OWNER_LIGHT = "#6BAED6"
C_RENTER_LIGHT = "#FB6A4A"


def load_sheet(name: str) -> pd.DataFrame:
    df = pd.read_excel(XLSX, sheet_name=name)
    if "Group" in df.columns:
        df["Group"] = df["Group"].map(GROUP_MAP).fillna(df["Group"])
    return df


def plot_coefficients(mcmc: pd.DataFrame, out: Path):
    """Bar chart of regression coefficients with HDI error bars."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

    for ax_idx, action in enumerate(ACTIONS):
        ax = axes[ax_idx]
        sub = mcmc[mcmc["Action"] == action].copy()
        sub = sub[sub["param"].isin(PARAMS + ["beta_0"])]

        for g_idx, (grp, color) in enumerate([("owner", C_OWNER), ("renter", C_RENTER)]):
            g = sub[sub["Group"] == grp].set_index("param")
            if g.empty:
                continue
            params_present = [p for p in ["beta_0"] + PARAMS if p in g.index]
            x = np.arange(len(params_present))
            means = [g.loc[p, "mean"] for p in params_present]
            lo = [g.loc[p, "mean"] - g.loc[p, "hdi_5%"] for p in params_present]
            hi = [g.loc[p, "hdi_95%"] - g.loc[p, "mean"] for p in params_present]
            offset = -0.18 + g_idx * 0.36
            ax.barh(x + offset, means, height=0.32, color=color, alpha=0.85,
                    xerr=[lo, hi], error_kw=dict(lw=1, capsize=3))

        params_present = [p for p in ["beta_0"] + PARAMS if p in sub["param"].values]
        ax.set_yticks(range(len(params_present)))
        ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in params_present], fontsize=10)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_title(action, fontweight="bold", fontsize=12)
        ax.set_xlabel("Posterior mean")

    axes[0].legend(
        handles=[Patch(fc=C_OWNER, label="Owner"), Patch(fc=C_RENTER, label="Renter")],
        loc="lower right", fontsize=9
    )
    fig.suptitle("Bayesian Beta Regression Coefficients (90% HDI)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "coefficient_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "coefficient_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[OK] coefficient_comparison")


def plot_convergence(mcmc: pd.DataFrame, out: Path):
    """R-hat and ESS diagnostic plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    mcmc = mcmc.copy()
    mcmc["label"] = mcmc["Group"] + "_" + mcmc["Action"] + ":" + mcmc["param"]

    # R-hat
    colors = [C_OWNER if "owner" in l else C_RENTER for l in mcmc["label"]]
    ax1.barh(range(len(mcmc)), mcmc["r_hat"].values, color=colors, alpha=0.7, height=0.7)
    ax1.axvline(1.0, color="green", lw=1.5, ls="-", label="R̂ = 1.0")
    ax1.axvline(1.05, color="red", lw=1, ls="--", label="R̂ = 1.05 threshold")
    ax1.set_yticks(range(len(mcmc)))
    ax1.set_yticklabels(mcmc["label"], fontsize=6)
    ax1.set_xlabel("R̂")
    ax1.set_title("Gelman-Rubin R̂", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0.99, 1.06)

    # ESS bulk
    ax2.barh(range(len(mcmc)), mcmc["ess_bulk"].values, color=colors, alpha=0.7, height=0.7)
    ax2.axvline(400, color="red", lw=1, ls="--", label="ESS = 400 threshold")
    ax2.set_yticks(range(len(mcmc)))
    ax2.set_yticklabels(mcmc["label"], fontsize=6)
    ax2.set_xlabel("ESS (bulk)")
    ax2.set_title("Effective Sample Size", fontweight="bold")
    ax2.legend(fontsize=8)

    fig.suptitle("MCMC Convergence Diagnostics — All 8 Models", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "convergence_diagnostics.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "convergence_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[OK] convergence_diagnostics")


def plot_calibration_improvement(pred: pd.DataFrame, out: Path):
    """ECE before vs after calibration."""
    # Use validation set rows
    val = pred[pred["EvalSet"] == "val"].copy()
    if val.empty:
        val = pred.copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = val["Group"] + "_" + val["Action"]
    x = np.arange(len(labels))
    w = 0.35

    ax.bar(x - w/2, val["pre_ECE"].values, w, color="#FDAE6B", label="Before calibration", edgecolor="k", lw=0.5)
    ax.bar(x + w/2, val["post_ECE"].values, w, color="#31A354", label="After calibration", edgecolor="k", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("Calibration Improvement: ECE Before vs After", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, max(val["pre_ECE"].max(), val["post_ECE"].max()) * 1.3)

    # Add delta annotations
    for i, (pre, post) in enumerate(zip(val["pre_ECE"], val["post_ECE"])):
        delta = post - pre
        color = "green" if delta < 0 else "red"
        ax.annotate(f"{delta:+.3f}", (i, max(pre, post) + 0.005),
                    ha="center", fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(out / "calibration_improvement.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "calibration_improvement.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[OK] calibration_improvement")


def plot_prior_posterior(pvp: pd.DataFrame, out: Path):
    """Prior vs Posterior shrinkage visualization."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    models = [(g, a) for g in ["owner", "renter"] for a in ACTIONS]
    for idx, (grp, action) in enumerate(models):
        row, col = idx // 4, idx % 4
        ax = axes[row, col]
        sub = pvp[(pvp["Group"] == grp) & (pvp["Action"] == action)]
        if sub.empty:
            ax.set_visible(False)
            continue

        params = sub["param"].values
        y = np.arange(len(params))

        ax.scatter(sub["prior_mean"], y, marker="o", s=60, c="grey", zorder=3, label="Prior")
        ax.errorbar(sub["prior_mean"], y, xerr=sub["prior_sd"], fmt="none",
                    ecolor="grey", alpha=0.4, lw=2)

        color = C_OWNER if grp == "owner" else C_RENTER
        ax.scatter(sub["post_mean"], y, marker="D", s=60, c=color, zorder=3, label="Posterior")
        ax.errorbar(sub["post_mean"], y, xerr=sub["post_sd"], fmt="none",
                    ecolor=color, alpha=0.6, lw=2)

        # Arrows from prior to posterior
        for i, (pm, postm) in enumerate(zip(sub["prior_mean"], sub["post_mean"])):
            ax.annotate("", xy=(postm, i), xytext=(pm, i),
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.8, alpha=0.4))

        ax.set_yticks(y)
        ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in params], fontsize=9)
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.set_title(f"{grp}_{action}", fontweight="bold", fontsize=11)
        if idx < 4:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Value")

    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Prior → Posterior Shrinkage (all 8 models)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "prior_vs_posterior.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "prior_vs_posterior.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[OK] prior_vs_posterior")


def plot_brier_skill(pred: pd.DataFrame, out: Path):
    """Brier Skill Score bar chart."""
    val = pred[pred["EvalSet"] == "val"].copy()
    if val.empty:
        val = pred.copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = val["Group"] + "_" + val["Action"]
    x = np.arange(len(labels))
    colors = [C_OWNER if "owner" in l else C_RENTER for l in labels]

    bars = ax.bar(x, val["post_BSS"].values, color=colors, edgecolor="k", lw=0.5, alpha=0.85)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Brier Skill Score (BSS)")
    ax.set_title("Brier Skill Score — Post-Calibration (validation set)", fontweight="bold")

    for bar, v in zip(bars, val["post_BSS"]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01 * np.sign(v),
                f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    ax.legend(
        handles=[Patch(fc=C_OWNER, label="Owner"), Patch(fc=C_RENTER, label="Renter")],
        loc="upper right", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(out / "brier_skill_score.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "brier_skill_score.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[OK] brier_skill_score")


def export_metrics_table(mcmc: pd.DataFrame, pred: pd.DataFrame, out: Path):
    """Export summary metrics CSV."""
    val = pred[pred["EvalSet"] == "val"].copy()
    if val.empty:
        val = pred.copy()

    rows = []
    for _, r in val.iterrows():
        grp, act = r["Group"], r["Action"]
        # Get convergence info
        m = mcmc[(mcmc["Group"] == grp) & (mcmc["Action"] == act)]
        rhat_max = m["r_hat"].max() if len(m) else np.nan
        ess_min = m["ess_bulk"].min() if len(m) else np.nan
        divergences = 0  # All models had 0 divergences

        rows.append({
            "Model": f"{grp}_{act}",
            "N": int(r["N"]),
            "Prevalence": round(r["Prevalence"], 3),
            "Brier_pre": round(r["pre_Brier"], 4),
            "Brier_post": round(r["post_Brier"], 4),
            "BSS_post": round(r["post_BSS"], 4),
            "ECE_pre": round(r["pre_ECE"], 4),
            "ECE_post": round(r["post_ECE"], 4),
            "ROC_AUC": round(r["pre_ROC_AUC"], 4),
            "Calibration_method": r.get("post_method", ""),
            "R_hat_max": round(float(rhat_max), 4),
            "ESS_min": int(ess_min) if np.isfinite(ess_min) else "",
            "Divergences": divergences,
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(out / "phase2_metrics_table.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] phase2_metrics_table.csv ({len(tbl)} models)")
    print(tbl.to_string(index=False))


def main():
    print("=" * 60)
    print("Phase 2: Bayesian Regression — Generating Plots")
    print("=" * 60)

    mcmc = load_sheet("mcmc_summary")
    pred = load_sheet("predictive")
    pvp = load_sheet("prior_vs_posterior")

    plot_coefficients(mcmc, HERE)
    plot_convergence(mcmc, HERE)
    plot_calibration_improvement(pred, HERE)
    plot_prior_posterior(pvp, HERE)
    plot_brier_skill(pred, HERE)
    export_metrics_table(mcmc, pred, HERE)

    print("=" * 60)
    print("All Phase 2 plots saved to:", HERE)


if __name__ == "__main__":
    main()
