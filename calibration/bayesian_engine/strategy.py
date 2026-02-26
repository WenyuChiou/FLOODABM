import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from .config import CFG
from .utils import _safe_clip, _logit, _inv_logit, setup_logger
from .metrics import _compute_metrics
from .calibration import Calibrators, _should_apply_cal
from .model import BayesianBetaRegressionModel
from .evaluation import run_param_eval_and_save
from .data import _syn_weight, _flag_synthetic

logger = setup_logger(__name__)

def _mcmc_budget(strategy: str, n_real: int) -> dict:
    """
    Determines MCMC budget based on strategy and sample size.
    
    Strategy-specific budgets:
    - nested: For repeated CV (smaller per-fold samples)
    - holdout: For standard train/val/test split
    
    Sample size adjustments:
    - Very small (n<30): More warmup to ensure convergence
    - Small (n<120): Moderate settings
    - Large (n>=120): Full robust settings
    """
    if strategy == "nested":
        # Nested CV: Use 3 chains minimum for reliable R-hat
        # Increased from 2→3 chains for better convergence diagnostics
        if n_real < 30:
            # Very small samples in nested CV folds
            return dict(num_chains=3, num_warmup=2500, num_samples=600, chain_method="vectorized")
        else:
            # Standard nested CV settings
            return dict(num_chains=3, num_warmup=2000, num_samples=500, chain_method="vectorized")
    
    # Holdout strategy: Sample size-based budgets
    if n_real < 30:
        # Very small samples: Conservative settings
        return dict(num_chains=3, num_warmup=3000, num_samples=1000, chain_method="vectorized")
    elif n_real < 120:
        # Small-to-medium samples
        return dict(num_chains=3, num_warmup=2500, num_samples=800, chain_method="vectorized")
    else:
        # Large samples: Full robust settings
        return dict(num_chains=4, num_warmup=4000, num_samples=1200, chain_method="vectorized")

def _pick_best_method_cv(rows: list[dict], is_rl: bool = False) -> str:
    """
    Select calibrator using CV rows.
    - Default: minimize mean Brier (with std penalty)
    - RL: add ECE weight; tie → lower ECE → prefer temp
    """
    if not rows:
        return "none"
    if is_rl:
        def score(r):
            return r.get("cv_mean_Brier", np.inf) \
                 + CFG.RL_STD_PENALTY * r.get("cv_std_Brier", 0.0) \
                 + CFG.RL_ECE_WEIGHT * r.get("cv_mean_ECE", 0.0)
        rows_sorted = sorted(rows, key=score)
        best = rows_sorted[0]
        tied = [r for r in rows_sorted
                if abs(r.get("cv_mean_Brier", np.inf) - best.get("cv_mean_Brier", np.inf)) <= CFG.RL_BRIER_TIE_EPS]
        if len(tied) > 1:
            tied_sorted = sorted(tied, key=lambda r: (r.get("cv_mean_ECE", np.inf),
                                                      0 if r["method"]=="temp" else 1))
            return tied_sorted[0]["method"]
        return best["method"]
    # non-RL
    return min(rows, key=lambda r: r.get("cv_mean_Brier", np.inf)
                          + CFG.RL_STD_PENALTY * r.get("cv_std_Brier", 0.0))["method"]

def _choose_strategy(n_real: int) -> str:
    return "nested" if n_real < CFG.SMALL_NESTED_THRESHOLD else "holdout"

def _strategy_for_action(action: str, n_real: int) -> str:
    base = _choose_strategy(n_real)
    if action != "RL":
        return base
    mode = getattr(CFG, "RL_EVAL_MODE", "auto").lower()
    if mode == "nested":
        return "nested"
    if mode == "holdout":
        return "holdout"
    return base  # auto

def prior_shift_correct(p: np.ndarray, pi_train: float, pi_target: float, eps: float = 1e-6) -> np.ndarray:
    p = _safe_clip(p, eps)
    def _L(x):
        x = np.clip(float(x), eps, 1-eps)
        return np.log(x) - np.log(1-x)
    delta = _L(pi_target) - _L(pi_train)
    return _safe_clip(_inv_logit(_logit(p) + delta), eps)

def _prior_shift_too_big(pi_a: float, pi_b: float, thr: float = 0.10) -> bool:
    return abs(float(pi_a) - float(pi_b)) > float(thr)

def rs_nested_cv(df_all, action, gname, adir,
                 outer_folds=5, outer_repeats=20, inner_folds=2,
                 n_bins=6, methods=("temp","platt"),
                 run_param_eval_once=True):
    """
    Repeated Stratified K-Fold (outer) with inner-CV for calibrator selection.
    Synthetic rows are used only in training (down-weighted); outer test is REAL only.

    Returns:
      - oof_pre_metrics, oof_post_metrics (pooled across all outer folds×repeats)
      - calib_summary_rows: per-method mean/std of inner-CV results across outer runs
      - best_method (by Brier+ECE scoring)
      - mean_tau (if temp appears), else np.nan
    """
    d = df_all.copy()
    d["is_syn"] = _flag_synthetic(d, "Source")
    d_real = d[~d["is_syn"]].reset_index(drop=True)
    d_syn  = d[ d["is_syn"]].reset_index(drop=True)

    if len(d_real) < max(outer_folds, inner_folds) + 1:
        outer_folds = max(2, min(len(d_real), outer_folds))
        inner_folds = max(2, min(len(d_real) - 1, inner_folds))

    y_all  = (d_real[action].values > 0.5).astype(int)

    rng_base = int(CFG.NP_SEED)

    y_pool, pre_pool, post_pool = [], [], []
    inner_cv_records = []
    mean_tau_list = []
    did_param_eval = False

    for rep in range(int(outer_repeats)):
        skf = StratifiedKFold(n_splits=int(outer_folds), shuffle=True, random_state=rng_base + rep)
        for fold_id, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all)):
            train_real = d_real.iloc[tr_idx].reset_index(drop=True)
            test_real  = d_real.iloc[te_idx].reset_index(drop=True)

            # training = outer-train(REAL) + ALL SYN (down-weighted)
            train_df = pd.concat([train_real, d_syn], ignore_index=True)
            is_syn_tr = train_df["is_syn"].to_numpy(bool)
            n_tr_real = (~is_syn_tr).sum(); n_tr_syn = is_syn_tr.sum()
            w_syn = _syn_weight(n_tr_real, n_tr_syn)
            W_tr = np.where(is_syn_tr, w_syn, 1.0)

            budget = _mcmc_budget(strategy="nested", n_real=len(train_real))
            logger.info(f"[MCMC-Budget] {gname}:{action} nested outer rep={rep} fold={fold_id} "
                  f"chains={budget['num_chains']} warmup={budget['num_warmup']} samples={budget['num_samples']}")

            # fit model
            model = BayesianBetaRegressionModel(action, gname)
            model.fit(train_df["TP_mean"].values, train_df["CP_mean"].values, train_df["SP_mean"].values,
                      train_df[action].values, W=W_tr, **budget)

            if run_param_eval_once and (not did_param_eval) and CFG.PARAM_EVAL_EARLY:
                run_param_eval_and_save(model=model, action=action, group=gname, out_dir=adir,
                                        hdi_prob=CFG.HDI, rope=CFG.ROPE_WIDTH)
                did_param_eval = True

            # inner-CV on outer-train REAL → 選校準法
            p_tr_pre_real = model.predict_proba_mu(train_real["TP_mean"], train_real["CP_mean"], train_real["SP_mean"])
            y_tr_bin_real = (train_real[action].values > 0.5).astype(int)
            tau_grid = CFG.RL_TAU_GRID if action=="RL" else CFG.TAU_GRID
            plattC   = CFG.RL_PLATT_C  if action=="RL" else 1.0

            inner_rows = Calibrators.evaluate_all_cv(
                p_tr_pre_real, y_tr_bin_real, n_bins=n_bins, folds=int(inner_folds),
                methods=tuple(methods), tau_grid=tau_grid, platt_C=plattC
            )
            for r in inner_rows:
                inner_cv_records.append({"rep": rep, "outer_fold": int(fold_id), **r})

            best_name = _pick_best_method_cv(inner_rows, is_rl=(action == "RL"))

            # 用 outer-train REAL 擬合選到的校準器
            if best_name == "temp":
                fit_cal = Calibrators.temp(p_tr_pre_real, y_tr_bin_real, tau_grid=tau_grid)
            elif best_name == "platt":
                fit_cal = Calibrators.platt(p_tr_pre_real, y_tr_bin_real, C=plattC)
            elif best_name == "isotonic":
                fit_cal = Calibrators.isotonic(p_tr_pre_real, y_tr_bin_real)
            elif best_name == "beta":
                fit_cal = Calibrators.beta(p_tr_pre_real, y_tr_bin_real)
            else:
                fit_cal = {"method": "none", "params": {}}

            if best_name == "temp" and "tau" in fit_cal.get("params", {}):
                mean_tau_list.append(float(fit_cal["params"]["tau"]))

            # 外層測試（REAL only）
            p_te_pre  = model.predict_proba_mu(test_real["TP_mean"], test_real["CP_mean"], test_real["SP_mean"])
            from .calibration import Calibrator # Avoid circular import at top level if possible, or just use it
            # Actually Calibrator is imported from .calibration
            p_te_post = Calibrator(method=fit_cal.get("method", "none"), params=fit_cal.get("params", {})).transform(p_te_pre)
            y_te = (test_real[action].values > 0.5).astype(int)

            if (action == "RL"):
                pi_cal = float(y_tr_bin_real.mean())
                pi_tgt = float(y_te.mean())
                p_te_post = prior_shift_correct(p_te_post, pi_cal, pi_tgt)

            # Gate：若 post 沒改善（Brier / ECE 門檻），就回退
            pre_m  = _compute_metrics(y_te, p_te_pre,  n_bins=n_bins)
            post_m = _compute_metrics(y_te, p_te_post, n_bins=n_bins)
            if not _should_apply_cal(pre_m, post_m,
                                     min_brier_gain=float(getattr(CFG, "MIN_BRIER_GAIN", 1e-3)),
                                     allow_ece_worsen=float(getattr(CFG, "ALLOW_ECE_WORSEN", 0.0))):
                p_te_post = p_te_pre

            y_pool.append(y_te)
            pre_pool.append(p_te_pre)
            post_pool.append(p_te_post)

    y_pool   = np.concatenate(y_pool)
    pre_pool = np.concatenate(pre_pool)
    post_pool= np.concatenate(post_pool)

    oof_pre_metrics  = _compute_metrics(y_pool, pre_pool,  n_bins=n_bins)
    oof_post_metrics = _compute_metrics(y_pool, post_pool, n_bins=n_bins)

    inner_df = pd.DataFrame(inner_cv_records) if inner_cv_records else pd.DataFrame()
    calib_summary_rows = []
    best_method = "none"
    if not inner_df.empty:
        for m in sorted(inner_df["method"].unique()):
            sub = inner_df[inner_df["method"] == m]
            calib_summary_rows.append({
                "method": m,
                "cv_mean_Brier": float(sub["cv_mean_Brier"].mean()),
                "cv_std_Brier":  float(sub["cv_mean_Brier"].std(ddof=1)) if len(sub)>1 else 0.0,
                "cv_mean_LogLoss": float(sub["cv_mean_LogLoss"].mean()),
                "cv_std_LogLoss":  float(sub["cv_mean_LogLoss"].std(ddof=1)) if len(sub)>1 else 0.0,
                "cv_mean_ECE": float(sub["cv_mean_ECE"].mean()),
                "cv_std_ECE":  float(sub["cv_mean_ECE"].std(ddof=1)) if len(sub)>1 else 0.0,
                "folds": int(inner_folds),
                "outer_folds": int(outer_folds),
                "repeats": int(outer_repeats),
            })
        best_method = _pick_best_method_cv(
            [{"method": r["method"], "cv_mean_Brier": r["cv_mean_Brier"],
              "cv_std_Brier": r["cv_std_Brier"], "cv_mean_ECE": r["cv_mean_ECE"]} for r in calib_summary_rows],
            is_rl=(action == "RL")
        )

    mean_tau = (float(np.mean(mean_tau_list)) if mean_tau_list else np.nan)

    return {
        "oof_pre_metrics": oof_pre_metrics,
        "oof_post_metrics": oof_post_metrics,
        "calib_summary_rows": calib_summary_rows,
        "best_method": best_method,
        "mean_tau": mean_tau
    }
