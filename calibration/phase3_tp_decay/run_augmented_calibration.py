# -*- coding: utf-8 -*-
"""
Augmented TP Decay Calibration — Owner/Renter
==============================================
Uses ALL survey respondents (not just flood-experienced).

Model change:
  Original:  TP(t) = TP₀ · exp(-ln2 · w_eff · Eff(t))
  Augmented: TP(t) = TP_base + (TP₀ - TP_base) · exp(-ln2 · w_eff · Eff(t))

Non-flood respondents treated as t → ∞ observations where TP → TP_base.
All other equations (τ(t), Eff(t), w_eff) remain IDENTICAL.

Outputs:
  - augmented_outputs/ directory with plots, tables, comparison
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.optimize import differential_evolution, minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
AUG_DATA   = SCRIPT_DIR / "augmented_cal_data.xlsx"
ORIG_PARAMS = SCRIPT_DIR / "outputs" / "tp_decay_params_for_abm.json"
OUTDIR     = SCRIPT_DIR / "augmented_outputs"

LN2 = np.log(2.0)

# =========================================================
# Core Math (identical to original model except TP formula)
# =========================================================

def tau_sat(t, tau0, tau_inf, k):
    t = np.asarray(t, dtype=float)
    return tau_inf - (tau_inf - tau0) * np.exp(-k * np.clip(t, 0.0, None))

def eff_closed_form(t, tau0, tau_inf, k):
    t = np.asarray(t, dtype=float)
    tau_diff = tau_inf - tau0
    num = tau_inf - tau_diff * np.exp(-k * t)
    num = np.maximum(num, 1e-12)
    tau0_safe = max(tau0, 1e-12)
    term1 = t / tau_inf
    term2 = (1.0 / (k * tau_inf)) * np.log(num / tau0_safe)
    return term1 + term2


def auto01(s):
    """Normalize series to [0,1] if it looks like a 1-5 scale."""
    vals = pd.to_numeric(s, errors="coerce").values.astype(float)
    if np.nanmax(vals) > 1.5:
        vals = vals / 5.0
    return np.clip(vals, 0, 1)


# =========================================================
# Augmented Fitting: TP(t) = TP_base + (TP0 - TP_base) * g(t)
# =========================================================

def fit_augmented(flood_df, no_flood_df, a, b,
                  r_target=0.5, lambda_pen=0.3,
                  bounds_tau0=(1.0, 12.0),
                  bounds_d=(4.0, 50.0),
                  bounds_k=(0.01, 0.6)):
    """
    Fit augmented model using flood + no-flood data.

    Flood respondents: TP(t) = TP_base + (TP0 - TP_base) * exp(-ln2 * w_eff * Eff(t))
    No-flood respondents: TP_obs ≈ TP_base (t → ∞)

    TP_base is estimated from no-flood respondents.
    TP0 is profiled out analytically given TP_base.
    """
    # Extract flood data
    PA_f = flood_df["PA"].values
    SC_f = flood_df["SC"].values
    TP_f = flood_df["TP"].values
    times_f = 0.5 * (flood_df["t_lo"].values + flood_df["t_hi"].values)
    w_eff_f = a * (1.0 - PA_f) + b * SC_f
    n_flood = len(flood_df)

    # TP_base from no-flood respondents
    TP_no_flood = no_flood_df["TP"].values
    tp_base = float(np.mean(TP_no_flood))
    n_no_flood = len(no_flood_df)

    # Total weight: flood observations contribute to decay fit,
    # no-flood contribute to TP_base estimation
    # Weight: n_no_flood observations anchor TP_base, n_flood fit the curve
    w_base = n_no_flood / (n_flood + n_no_flood)  # relative weight of baseline term
    w_decay = n_flood / (n_flood + n_no_flood)

    bounds = [bounds_tau0, bounds_d, bounds_k]

    def objective(x):
        t0, dlt, k_val = x
        t_inf = t0 + dlt

        # Decay basis for flood respondents
        Eff_f = eff_closed_form(times_f, t0, t_inf, k_val)
        g_f = np.exp(-LN2 * w_eff_f * Eff_f)

        # Profile out TP0:
        # TP_f ≈ tp_base + (TP0 - tp_base) * g_f
        # Let delta = TP0 - tp_base
        # TP_f - tp_base ≈ delta * g_f
        # delta_hat = sum((TP_f - tp_base) * g_f) / sum(g_f^2)
        residuals_from_base = TP_f - tp_base
        delta_hat = np.sum(residuals_from_base * g_f) / (np.sum(g_f * g_f) + 1e-12)
        delta_hat = max(delta_hat, 0.01)  # TP0 > TP_base

        # Predictions for flood respondents
        y_pred_f = tp_base + delta_hat * g_f

        # MSE on flood data (primary objective)
        mse_flood = np.mean((TP_f - y_pred_f)**2)

        # MSE on no-flood data (TP_base fit)
        mse_base = np.mean((TP_no_flood - tp_base)**2)

        # Combined loss (weighted)
        loss = w_decay * mse_flood + w_base * mse_base

        # Penalty on 5-year ratio
        eff5 = eff_closed_form(5.0, t0, t_inf, k_val)
        w_ref = a * 0.5 + b * 0.5
        g5 = np.exp(-LN2 * w_ref * eff5)
        pred_ratio_5 = (tp_base + delta_hat * g5) / (tp_base + delta_hat)
        loss += lambda_pen * (pred_ratio_5 - r_target)**2

        return loss

    res = differential_evolution(objective, bounds, strategy='best1bin',
                                 popsize=15, mutation=(0.5, 1.0),
                                 recombination=0.7, seed=42, maxiter=300)

    best_t0, best_delta, best_k = res.x
    best_t_inf = best_t0 + best_delta

    # Final estimates
    Eff_f = eff_closed_form(times_f, best_t0, best_t_inf, best_k)
    g_f = np.exp(-LN2 * w_eff_f * Eff_f)
    residuals_from_base = TP_f - tp_base
    delta_hat = max(np.sum(residuals_from_base * g_f) / (np.sum(g_f * g_f) + 1e-12), 0.01)
    tp0_hat = tp_base + delta_hat

    y_pred_f = tp_base + delta_hat * g_f
    mse_flood = np.mean((TP_f - y_pred_f)**2)
    mae_flood = np.mean(np.abs(TP_f - y_pred_f))

    # Combined RMSE (all data)
    y_pred_all = np.concatenate([y_pred_f, np.full(n_no_flood, tp_base)])
    y_obs_all = np.concatenate([TP_f, TP_no_flood])
    rmse_all = np.sqrt(np.mean((y_obs_all - y_pred_all)**2))

    def get_ratio(yr):
        e = eff_closed_form(yr, best_t0, best_t_inf, best_k)
        w_mean = a * np.mean(1 - PA_f) + b * np.mean(SC_f)
        tp_at_yr = tp_base + delta_hat * np.exp(-LN2 * w_mean * e)
        return float(tp_at_yr / tp0_hat)

    return dict(
        alpha=a, beta=b,
        tau0=best_t0, tau_inf=best_t_inf, k=best_k,
        TP0_hat=float(tp0_hat), TP_base=float(tp_base),
        RMSE_flood=np.sqrt(mse_flood), MAE_flood=mae_flood,
        RMSE_all=rmse_all,
        n_flood=n_flood, n_no_flood=n_no_flood,
        PenalizedObj=res.fun,
        TP5_TP0=get_ratio(5.0),
        TP10_TP0=get_ratio(10.0),
        TP15_TP0=get_ratio(15.0),
    )


# =========================================================
# Grid Search
# =========================================================

def run_augmented_grid(flood_df, no_flood_df, group_name, conf, n_jobs=1):
    """Run grid search over (alpha, beta) for augmented model."""
    from joblib import Parallel, delayed

    alphas = np.linspace(conf["alpha_range"][0], conf["alpha_range"][1], conf["alpha_steps"])
    betas  = np.linspace(conf["beta_range"][0],  conf["beta_range"][1],  conf["beta_steps"])
    grid_points = [(a, b) for a in alphas for b in betas]
    print(f"[{group_name}] Augmented Grid Search ({len(alphas)}x{len(betas)})...")

    def fit_point(a, b):
        return fit_augmented(flood_df, no_flood_df, a, b,
                             r_target=conf["r_target"],
                             lambda_pen=conf["lambda_pen"],
                             bounds_tau0=conf["bounds_tau0"],
                             bounds_d=conf["bounds_d"],
                             bounds_k=conf["bounds_k"])

    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_point)(a, b) for a, b in grid_points
    )
    for res, (a, b) in zip(results, grid_points):
        res["Group"] = group_name
    return pd.DataFrame(results)


# =========================================================
# Joint Selection + Fine-Tuning
# =========================================================

CROSS_CONSTRAINTS = dict(min_gap_alpha=0.1, min_gap_beta=0.1)

def select_best_joint(owner_grid, renter_grid):
    cross = CROSS_CONSTRAINTS
    def get_cands(df):
        min_rmse = df["RMSE_flood"].min()
        return df[df["RMSE_flood"] <= min_rmse * 1.5]

    oc = get_cands(owner_grid)
    rc = get_cands(renter_grid)

    pairs = []
    for _, r1 in oc.iterrows():
        for _, r2 in rc.iterrows():
            if abs(r1["alpha"] - r2["alpha"]) < cross["min_gap_alpha"]: continue
            if abs(r1["beta"]  - r2["beta"])  < cross["min_gap_beta"]:  continue
            score = r1["RMSE_flood"] + r2["RMSE_flood"]
            pairs.append({"score": score, "owner": r1, "renter": r2,
                          "gap_alpha": abs(r1["alpha"] - r2["alpha"]),
                          "gap_beta": abs(r1["beta"] - r2["beta"])})

    if not pairs:
        print("WARNING: No pairs satisfy constraints. Relaxing gap to 0.05.")
        for _, r1 in oc.iterrows():
            for _, r2 in rc.iterrows():
                if abs(r1["alpha"] - r2["alpha"]) < 0.05: continue
                if abs(r1["beta"]  - r2["beta"])  < 0.05: continue
                score = r1["RMSE_flood"] + r2["RMSE_flood"]
                pairs.append({"score": score, "owner": r1, "renter": r2,
                              "gap_alpha": abs(r1["alpha"] - r2["alpha"]),
                              "gap_beta": abs(r1["beta"] - r2["beta"])})

    if not pairs:
        raise ValueError("No valid pairs found even with relaxed constraints")

    pairs.sort(key=lambda x: x["score"])
    return pairs[0]


# =========================================================
# Bootstrap
# =========================================================

def run_bootstrap_augmented(flood_df, no_flood_df, alpha, beta, group_name, conf, n_boot=100):
    from joblib import Parallel, delayed
    print(f"[{group_name}] Bootstrap ({n_boot} samples)...")

    def boot_iter(seed):
        rng = np.random.default_rng(seed)
        flood_boot = flood_df.sample(n=len(flood_df), replace=True, random_state=seed)
        nf_boot = no_flood_df.sample(n=len(no_flood_df), replace=True, random_state=seed+10000)
        return fit_augmented(flood_boot, nf_boot, alpha, beta,
                             r_target=conf["r_target"],
                             lambda_pen=conf["lambda_pen"],
                             bounds_tau0=conf["bounds_tau0"],
                             bounds_d=conf["bounds_d"],
                             bounds_k=conf["bounds_k"])

    seeds = np.random.randint(0, 100000, size=n_boot)
    results = Parallel(n_jobs=-1)(delayed(boot_iter)(s) for s in seeds)
    return pd.DataFrame(results)


# =========================================================
# Visualization
# =========================================================

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

COLORS = {
    "owner": "#2171B5", "owner_fill": "#6BAED6",
    "renter": "#CB181D", "renter_fill": "#FB6A4A",
    "orig_owner": "#2171B5", "orig_renter": "#CB181D",
}


def plot_augmented_curves(best, flood_data, no_flood_data, outdir, orig_params=None):
    """Plot decay curves with TP_base floor, optionally comparing to original model."""
    plt.rcParams.update(STYLE)
    t = np.linspace(0, 40, 801)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for idx, grp in enumerate(["owner", "renter"]):
        ax = axes[idx]
        row = best[grp]
        a, b = row["alpha"], row["beta"]
        t0, t_inf, k = row["tau0"], row["tau_inf"], row["k"]
        tp_base = row["TP_base"]
        tp0 = row["TP0_hat"]

        # Mean PA, SC from flood data
        f_df = flood_data[grp]
        PA_bar = f_df["PA"].mean()
        SC_bar = f_df["SC"].mean()
        w_eff = a * (1 - PA_bar) + b * SC_bar

        Eff = eff_closed_form(t, t0, t_inf, k)
        g = np.exp(-LN2 * w_eff * Eff)
        TP_aug = tp_base + (tp0 - tp_base) * g

        ax.plot(t, TP_aug, color=COLORS[grp], linewidth=2.5,
                label=f"Augmented (RMSE={row['RMSE_flood']:.3f})")

        # TP_base floor
        ax.axhline(tp_base, color=COLORS[grp], linestyle=":", alpha=0.6,
                    label=f"TP_base = {tp_base:.3f}")

        # Original model (if available)
        if orig_params and grp in orig_params:
            op = orig_params[grp]
            Eff_o = eff_closed_form(t, op["tau0"], op["tau_inf"], op["k"])
            w_eff_o = op["alpha"] * (1 - PA_bar) + op["beta"] * SC_bar
            TP_orig = np.exp(-LN2 * w_eff_o * Eff_o)
            # Scale to 0-1 range (original model doesn't have explicit TP0)
            ax.plot(t, TP_orig, color="gray", linewidth=1.5, linestyle="--",
                    alpha=0.7, label=f"Original (RMSE={op.get('RMSE', '?')})")

        # Scatter flood observations
        times_f = 0.5 * (f_df["t_lo"].values + f_df["t_hi"].values)
        ax.scatter(times_f, f_df["TP"].values, color=COLORS[grp],
                   alpha=0.3, s=15, zorder=5, label=f"Flood obs (n={len(f_df)})")

        # Scatter no-flood as points at x=35-40
        nf_df = no_flood_data[grp]
        nf_x = np.random.default_rng(42).uniform(35, 40, len(nf_df))
        ax.scatter(nf_x, nf_df["TP"].values, color=COLORS[f"{grp}_fill"],
                   alpha=0.15, s=8, marker="x", zorder=4,
                   label=f"No-flood (n={len(nf_df)})")

        ax.set_title(f"{grp.capitalize()} (n={row['n_flood']}+{row['n_no_flood']})")
        ax.set_xlabel("Years since Flood")
        if idx == 0:
            ax.set_ylabel("Threat Perception (0-1 scale)")
        ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.9)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 42)

    fig.suptitle("Augmented TP Decay Model: TP(t) = TP_base + (TP₀ − TP_base) · g(t)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "augmented_decay_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("[OK] augmented_decay_curves.png")


def plot_residuals_augmented(best, flood_data, no_flood_data, outdir):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for idx, grp in enumerate(["owner", "renter"]):
        ax = axes[idx]
        row = best[grp]
        a, b = row["alpha"], row["beta"]
        t0, t_inf, k = row["tau0"], row["tau_inf"], row["k"]
        tp_base = row["TP_base"]
        tp0 = row["TP0_hat"]

        f_df = flood_data[grp]
        PA_f = f_df["PA"].values
        SC_f = f_df["SC"].values
        TP_f = f_df["TP"].values
        times_f = 0.5 * (f_df["t_lo"].values + f_df["t_hi"].values)
        w_eff = a * (1 - PA_f) + b * SC_f
        Eff = eff_closed_form(times_f, t0, t_inf, k)
        g = np.exp(-LN2 * w_eff * Eff)
        pred_f = tp_base + (tp0 - tp_base) * g

        nf_df = no_flood_data[grp]
        pred_nf = np.full(len(nf_df), tp_base)

        ax.scatter(TP_f, pred_f, color=COLORS[grp], alpha=0.5, s=20,
                   label=f"Flood (n={len(f_df)})", edgecolors='w', linewidth=0.3)
        ax.scatter(nf_df["TP"].values, pred_nf, color=COLORS[f"{grp}_fill"],
                   alpha=0.2, s=10, marker="x", label=f"No-flood (n={len(nf_df)})")

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
        ax.set_xlabel("Observed TP")
        ax.set_ylabel("Predicted TP")
        ax.set_title(f"{grp.capitalize()} — Obs vs Pred")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

        stats = f"RMSE(flood)={row['RMSE_flood']:.3f}\nRMSE(all)={row['RMSE_all']:.3f}"
        ax.text(0.95, 0.05, stats, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    fig.suptitle("Augmented Model: Observed vs Predicted", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "augmented_residuals.png", bbox_inches="tight")
    plt.close(fig)
    print("[OK] augmented_residuals.png")


def plot_comparison_table(best, orig_params, outdir):
    """Create a comparison table figure: original vs augmented parameters."""
    plt.rcParams.update(STYLE)

    # Build comparison data
    rows = []
    for grp in ["owner", "renter"]:
        aug = best[grp]
        orig = orig_params.get(grp, {}) if orig_params else {}
        rows.append({
            "Group": grp.capitalize(),
            "Model": "Original",
            "alpha": orig.get("alpha", "-"),
            "beta": orig.get("beta", "-"),
            "tau0": orig.get("tau0", "-"),
            "tau_inf": orig.get("tau_inf", "-"),
            "k": orig.get("k", "-"),
            "TP_base": "-",
            "TP0_hat": "-",
            "RMSE": orig.get("RMSE", "-"),
            "n": f"{aug['n_flood']}",
        })
        rows.append({
            "Group": grp.capitalize(),
            "Model": "Augmented",
            "alpha": f"{aug['alpha']:.4f}",
            "beta": f"{aug['beta']:.4f}",
            "tau0": f"{aug['tau0']:.2f}",
            "tau_inf": f"{aug['tau_inf']:.2f}",
            "k": f"{aug['k']:.4f}",
            "TP_base": f"{aug['TP_base']:.4f}",
            "TP0_hat": f"{aug['TP0_hat']:.4f}",
            "RMSE": f"{aug['RMSE_flood']:.4f}",
            "n": f"{aug['n_flood']}+{aug['n_no_flood']}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "comparison_table.csv", index=False, encoding="utf-8-sig")
    print("[OK] comparison_table.csv")
    return df


def plot_ci_augmented(best, owner_boot, renter_boot, flood_data, no_flood_data, outdir):
    """Bootstrap confidence interval ribbons for augmented model."""
    plt.rcParams.update(STYLE)
    t = np.linspace(0, 40, 801)
    fig, ax = plt.subplots(figsize=(7, 5))

    for grp, boot_df, color, fill in [
        ("owner", owner_boot, COLORS["owner"], COLORS["owner_fill"]),
        ("renter", renter_boot, COLORS["renter"], COLORS["renter_fill"]),
    ]:
        row = best[grp]
        a, b = row["alpha"], row["beta"]
        f_df = flood_data[grp]
        PA_bar, SC_bar = f_df["PA"].mean(), f_df["SC"].mean()
        w_eff = a * (1 - PA_bar) + b * SC_bar

        curves = []
        for _, sample in boot_df.iterrows():
            tp_base = sample["TP_base"]
            tp0 = sample["TP0_hat"]
            Eff = eff_closed_form(t, sample["tau0"], sample["tau_inf"], sample["k"])
            g = np.exp(-LN2 * w_eff * Eff)
            curves.append(tp_base + (tp0 - tp_base) * g)
        curves = np.array(curves)

        lo = np.percentile(curves, 2.5, axis=0)
        hi = np.percentile(curves, 97.5, axis=0)
        med = np.percentile(curves, 50.0, axis=0)

        ax.fill_between(t, lo, hi, color=fill, alpha=0.4, label=f"{grp.capitalize()} 95% CI")
        ax.plot(t, med, color=color, label=f"{grp.capitalize()} Median")

    ax.set_title("Augmented Model: Bootstrap Parameter Uncertainty (N=100)")
    ax.set_xlabel("Years since Flood")
    ax.set_ylabel("Threat Perception (0-1)")
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(outdir / "augmented_confidence_intervals.png")
    plt.close(fig)
    print("[OK] augmented_confidence_intervals.png")


# =========================================================
# Main
# =========================================================

GRID_SETTINGS = dict(
    owner=dict(
        alpha_range=(0.10, 0.60), alpha_steps=20,
        beta_range=(0.10, 0.60),  beta_steps=20,
        r_target=0.5, lambda_pen=0.30,
        bounds_tau0=(1.0, 12.0), bounds_d=(4.0, 50.0), bounds_k=(0.01, 0.6),
    ),
    renter=dict(
        alpha_range=(0.10, 0.60), alpha_steps=20,
        beta_range=(0.10, 0.60),  beta_steps=20,
        r_target=0.5, lambda_pen=0.20,
        bounds_tau0=(1.0, 12.0), bounds_d=(4.0, 50.0), bounds_k=(0.01, 0.6),
    ),
)


def load_augmented_data(path, sheet):
    """Load augmented data and split into flood / no-flood."""
    df = pd.read_excel(path, sheet_name=sheet)

    PA = auto01(df["PA_mean"])
    SC = auto01(df["SC_mean"])
    TP = auto01(df["TP_mean"])
    has_flood = df["has_flood"].values.astype(int)
    q15_year = pd.to_numeric(df["Q15_year"], errors="coerce").values  # Q15_year = original Q12

    full = pd.DataFrame({"PA": PA, "SC": SC, "TP": TP,
                          "has_flood": has_flood, "t_lo": q15_year, "t_hi": q15_year})

    flood_mask = full["has_flood"] == 1
    flood_df = full[flood_mask].dropna(subset=["t_lo"]).copy()
    no_flood_df = full[~flood_mask].copy()

    return flood_df, no_flood_df


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Augmented TP Decay Calibration — Owner/Renter")
    print("=" * 60)

    # Load data
    flood_data, no_flood_data = {}, {}
    for grp in ["owner", "renter"]:
        f, nf = load_augmented_data(AUG_DATA, f"{grp}_augmented")
        flood_data[grp] = f
        no_flood_data[grp] = nf
        print(f"  {grp}: flood={len(f)}, no_flood={len(nf)}, "
              f"TP_flood={f['TP'].mean():.4f}, TP_base={nf['TP'].mean():.4f}")

    # Grid search
    print("\n--- Grid Search ---")
    grids = {}
    for grp in ["owner", "renter"]:
        conf = GRID_SETTINGS[grp]
        grids[grp] = run_augmented_grid(
            flood_data[grp], no_flood_data[grp], grp, conf, n_jobs=-1
        )
        best_row = grids[grp].loc[grids[grp]["RMSE_flood"].idxmin()]
        print(f"  {grp} best: alpha={best_row['alpha']:.4f}, beta={best_row['beta']:.4f}, "
              f"RMSE={best_row['RMSE_flood']:.4f}")

    # Joint selection
    print("\n--- Joint Selection ---")
    best_pair = select_best_joint(grids["owner"], grids["renter"])
    print(f"  Score (RMSE sum): {best_pair['score']:.4f}")

    # Package as dict
    best = {}
    for grp in ["owner", "renter"]:
        best[grp] = best_pair[grp].to_dict()

    print("\n>>> AUGMENTED MODEL RESULTS <<<")
    for grp in ["owner", "renter"]:
        r = best[grp]
        print(f"  {grp:>6s}: alpha={r['alpha']:.4f}, beta={r['beta']:.4f}, "
              f"tau0={r['tau0']:.2f}, tau_inf={r['tau_inf']:.2f}, k={r['k']:.4f}, "
              f"TP_base={r['TP_base']:.4f}, TP0_hat={r['TP0_hat']:.4f}, "
              f"RMSE(flood)={r['RMSE_flood']:.4f}, RMSE(all)={r['RMSE_all']:.4f}")

    # Bootstrap
    print("\n--- Bootstrap ---")
    boots = {}
    for grp in ["owner", "renter"]:
        conf = GRID_SETTINGS[grp]
        boots[grp] = run_bootstrap_augmented(
            flood_data[grp], no_flood_data[grp],
            best[grp]["alpha"], best[grp]["beta"],
            grp, conf, n_boot=100
        )

    # Load original params for comparison
    orig_params = None
    if ORIG_PARAMS.exists():
        with open(ORIG_PARAMS, "r") as f:
            orig_params = json.load(f)
        print(f"\nOriginal params loaded from: {ORIG_PARAMS}")

    # Plots
    print("\n--- Generating Plots ---")
    plot_augmented_curves(best, flood_data, no_flood_data, OUTDIR, orig_params)
    plot_residuals_augmented(best, flood_data, no_flood_data, OUTDIR)
    plot_ci_augmented(best, boots["owner"], boots["renter"], flood_data, no_flood_data, OUTDIR)
    comp_df = plot_comparison_table(best, orig_params, OUTDIR)

    # Export params for ABM
    params_for_abm = {}
    for grp in ["owner", "renter"]:
        r = best[grp]
        params_for_abm[grp] = {
            "alpha": round(float(r["alpha"]), 4),
            "beta": round(float(r["beta"]), 4),
            "tau0": round(float(r["tau0"]), 4),
            "tau_inf": round(float(r["tau_inf"]), 4),
            "k": round(float(r["k"]), 4),
            "TP_base": round(float(r["TP_base"]), 4),
            "TP0_hat": round(float(r["TP0_hat"]), 4),
            "RMSE_flood": round(float(r["RMSE_flood"]), 4),
            "RMSE_all": round(float(r["RMSE_all"]), 4),
        }

    params_path = OUTDIR / "augmented_tp_params_for_abm.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params_for_abm, f, indent=2)
    print(f"\n[OK] ABM params: {params_path}")

    # Summary CSV
    summary_rows = []
    for grp in ["owner", "renter"]:
        r = best[grp]
        summary_rows.append({k: round(v, 4) if isinstance(v, float) else v
                             for k, v in r.items()})
    pd.DataFrame(summary_rows).to_csv(
        OUTDIR / "augmented_metrics_table.csv", index=False, encoding="utf-8-sig")
    print("[OK] augmented_metrics_table.csv")

    # Print comparison
    if orig_params:
        print("\n" + "=" * 60)
        print("COMPARISON: Original vs Augmented")
        print("=" * 60)
        for grp in ["owner", "renter"]:
            r = best[grp]
            o = orig_params[grp]
            print(f"\n{grp.upper()}:")
            print(f"  {'Param':<12} {'Original':>10} {'Augmented':>10} {'Change':>10}")
            print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
            for p in ["alpha", "beta", "tau0", "tau_inf", "k"]:
                ov = o[p]
                av = r[p]
                ch = av - ov
                print(f"  {p:<12} {ov:>10.4f} {av:>10.4f} {ch:>+10.4f}")
            print(f"  {'TP_base':<12} {'N/A':>10} {r['TP_base']:>10.4f}")
            print(f"  {'TP0_hat':<12} {'N/A':>10} {r['TP0_hat']:>10.4f}")
            print(f"  {'RMSE(flood)':<12} {o['RMSE']:>10.4f} {r['RMSE_flood']:>10.4f}")
            print(f"  {'n_data':<12} {r['n_flood']:>10} {r['n_flood']+r['n_no_flood']:>10}")

    print("\n" + "=" * 60)
    print(f"All outputs in: {OUTDIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
