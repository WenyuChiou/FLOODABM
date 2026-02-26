# core/optimizer.py
# Calibration Algorithms (Parallel + Fine Tuning + Bootstrap) — Owner/Renter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from core.model import fit_tau_sat_given_ab, eff_closed_form, LN2

def calculate_bic_aic(n_samples, mse, k_params):
    if mse <= 1e-12: mse = 1e-12
    nll = n_samples * np.log(np.sqrt(mse)) + (n_samples/2) * (1 + np.log(2*np.pi))
    aic = 2 * k_params + 2 * nll
    bic = k_params * np.log(n_samples) + 2 * nll
    return aic, bic

def _fit_single_point(df, a, b, conf):
    return fit_tau_sat_given_ab(
        df, a, b,
        conf["r_target"], conf["lambda_pen"],
        n_q=cfg.COMPUTE["n_quantiles"],
        t_max=cfg.COMPUTE["t_max"],
        dt=cfg.COMPUTE["dt"],
        bounds_tau0=conf["bounds_tau0"],
        bounds_d=conf["bounds_d"],
        bounds_k=conf["bounds_k"]
    )

def run_grid_search(df: pd.DataFrame, group_name: str, n_jobs=-1) -> pd.DataFrame:
    conf = cfg.GRID_SETTINGS[group_name]
    alphas = np.linspace(conf["alpha_range"][0], conf["alpha_range"][1], conf["alpha_steps"])
    betas = np.linspace(conf["beta_range"][0],  conf["beta_range"][1],  conf["beta_steps"])

    print(f"[{group_name}] Parallel Grid Search ({len(alphas)}x{len(betas)})...")
    grid_points = [(a, b) for a in alphas for b in betas]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_single_point)(df, a, b, conf) for a, b in grid_points
    )

    for res, (a, b) in zip(results, grid_points):
        res["Group"] = group_name

    return pd.DataFrame(results)

def select_best_joint_pair(owner_df: pd.DataFrame, renter_df: pd.DataFrame) -> dict:
    cross = cfg.CROSS_CONSTRAINTS

    def get_candidates(df):
        min_rmse = df["RMSE"].min()
        return df[df["RMSE"] <= min_rmse * 1.5]

    owner_cand = get_candidates(owner_df)
    renter_cand = get_candidates(renter_df)

    valid_pairs = []
    for _, r1 in owner_cand.iterrows():
        for _, r2 in renter_cand.iterrows():
            if abs(r1["alpha"] - r2["alpha"]) < cross["min_gap_alpha"]: continue
            if abs(r1["beta"]  - r2["beta"])  < cross["min_gap_beta"]:  continue

            score = r1["RMSE"] + r2["RMSE"]
            valid_pairs.append({
                "score": score, "owner": r1, "renter": r2,
                "gap_alpha": abs(r1["alpha"] - r2["alpha"]),
                "gap_beta": abs(r1["beta"] - r2["beta"])
            })

    if not valid_pairs:
        raise ValueError("No pairs satisfied constraints.")

    valid_pairs.sort(key=lambda x: x["score"])
    return valid_pairs[0]

def fine_tune_best_pair(best_pair, owner_raw_df, renter_raw_df):
    print(">>> Fine-Tuning Parameters (Gradient Descent) <<<")

    def tune_one(df, start_row, conf):
        a0, b0 = start_row["alpha"], start_row["beta"]

        def loss(x):
            a, b = x
            res = _fit_single_point(df, a, b, conf)
            return res["RMSE"]

        bnds = [(0.01, 1.0), (0.01, 1.0)]
        opt = minimize(loss, [a0, b0], bounds=bnds, method='L-BFGS-B', options={'ftol': 1e-4})

        final_res = _fit_single_point(df, opt.x[0], opt.x[1], conf)
        final_res["Group"] = start_row["Group"]
        n = len(df)
        aic, bic = calculate_bic_aic(n, final_res["RMSE"]**2, 5)
        final_res["AIC"] = aic
        final_res["BIC"] = bic

        return pd.Series(final_res)

    owner_conf = cfg.GRID_SETTINGS["owner"]
    renter_conf = cfg.GRID_SETTINGS["renter"]

    new_owner = tune_one(owner_raw_df, best_pair["owner"], owner_conf)
    new_renter = tune_one(renter_raw_df, best_pair["renter"], renter_conf)

    cross = cfg.CROSS_CONSTRAINTS
    gap_a = abs(new_owner["alpha"] - new_renter["alpha"])
    gap_b = abs(new_owner["beta"] - new_renter["beta"])

    if gap_a < cross["min_gap_alpha"] or gap_b < cross["min_gap_beta"]:
        print("Warning: Fine-tuning violated constraints. Reverting to Grid Search result.")
        return best_pair

    print(f"Fine-tuned: RMSE Sum {best_pair['score']:.4f} -> {new_owner['RMSE']+new_renter['RMSE']:.4f}")
    return {"score": new_owner["RMSE"]+new_renter["RMSE"],
            "owner": new_owner, "renter": new_renter,
            "gap_alpha": gap_a, "gap_beta": gap_b}

def run_bootstrap(df: pd.DataFrame, alpha: float, beta: float, group_name: str, n_boot=200):
    print(f"[{group_name}] Running Bootstrap ({n_boot} samples)...")
    conf = cfg.GRID_SETTINGS[group_name]

    def boot_iter(seed):
        resampled_df = df.sample(n=len(df), replace=True, random_state=seed)
        res = _fit_single_point(resampled_df, alpha, beta, conf)
        return res

    seeds = np.random.randint(0, 100000, size=n_boot)
    results = Parallel(n_jobs=-1)(delayed(boot_iter)(s) for s in seeds)

    return pd.DataFrame(results)
