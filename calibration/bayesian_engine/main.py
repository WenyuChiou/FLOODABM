import os
import json
import numpy as np
import pandas as pd
import random as pyrandom

from .config import CFG
from .data import get_data, _flag_synthetic, _syn_weight
from .utils import _adaptive_bins, _calib_methods_by_n, _cv_folds_by_n, _safe_clip, setup_logger
from .model import BayesianBetaRegressionModel
from .calibration import Calibrators, Calibrator, _should_apply_cal
from .evaluation import run_param_eval_and_save
from .metrics import _compute_metrics
from .strategy import rs_nested_cv, _strategy_for_action, _mcmc_budget, _pick_best_method_cv, _prior_shift_too_big, prior_shift_correct

def main():
    logger = setup_logger("BayesianEngine_Main")
    
    np.random.seed(getattr(CFG, "NP_SEED", 0))
    pyrandom.seed(getattr(CFG, "PY_SEED", 0))

    os.makedirs(CFG.RESULTS_DIR, exist_ok=True)
    models_dir = os.path.join(CFG.RESULTS_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    data = get_data()

    all_pred_rows, all_calib_rows = [], []
    all_mcmc_rows, all_pq_rows, all_pp_rows, all_corr_rows = [], [], [], []

    for key, gname in CFG.GROUPS:
        if key not in data:
            logger.warning(f"[Skip] {key} not found in data dict.")
            continue

        gdir = os.path.join(CFG.RESULTS_DIR, f"{gname}_group")
        os.makedirs(gdir, exist_ok=True)

        df_full = data[key]
        for action in CFG.ACTIONS:
            if action not in df_full.columns:
                logger.warning(f"[Skip] {gname}:{action} not in dataframe.")
                continue

            d = df_full.dropna(subset=[action]).copy()
            if len(d) < 10:
                logger.warning(f"[Skip] {gname}:{action} too few rows (n={len(d)}).")
                continue

            adir = os.path.join(gdir, action)
            os.makedirs(adir, exist_ok=True)
            os.makedirs(adir, exist_ok=True)
            logger.info(f"\n=== {gname}:{action}  (n={len(d)}) ===")

            d["is_syn"] = _flag_synthetic(d, "Source")
            d_real = d[~d["is_syn"]].reset_index(drop=True)
            d_syn  = d[ d["is_syn"]].reset_index(drop=True)
            n_real_total = len(d_real)

            if n_real_total < 10:
                logger.warning(f"[Skip] {gname}:{action} real too few (n_real={n_real_total}).")
                continue

            N_BINS_THIS   = _adaptive_bins(n_real_total)
            CAL_METHODS   = tuple(_calib_methods_by_n(n_real_total))
            CV_FOLDS_THIS = int(_cv_folds_by_n(n_real_total))
            strategy = _strategy_for_action(action, n_real_total)
            CAL_METHODS = tuple(_calib_methods_by_n(n_real_total))

            logger.info(f"[Strategy] n_real={n_real_total}  → {strategy.upper()}  "
                  f"(bins={N_BINS_THIS}, calib={list(CAL_METHODS)}, innerCV={CV_FOLDS_THIS})")

            # ---- RL 專用覆寫 ----
            if action == "RL":
                if strategy == "nested":
                    N_BINS_THIS = min(N_BINS_THIS, 8)
                if CFG.RL_KEEP_RL_METHODS:
                    CAL_METHODS = tuple(CFG.RL_CALIB_METHODS)  # ["temp","platt"]

            if strategy == "nested":
                # ---------- RS-Nested CV ----------
                res = rs_nested_cv(
                    df_all=d, action=action, gname=gname, adir=adir,
                    outer_folds=max(2, min(CFG.OUTER_FOLDS_SMALL, n_real_total)),
                    outer_repeats=int(CFG.OUTER_REPEATS_SMALL),
                    inner_folds=int(CFG.INNER_FOLDS_SMALL),
                    n_bins=N_BINS_THIS, methods=CAL_METHODS,
                    run_param_eval_once=True
                )

                # 儲存 OOF metrics
                pre_row  = {"Action": action, "Group": gname, "Stage": "pre",  "EvalSet": "oof", **res["oof_pre_metrics"]}
                post_row = {"Action": action, "Group": gname, "Stage": "post", "EvalSet": "oof",
                            "method": res["best_method"], **res["oof_post_metrics"]}
                met_path = os.path.join(adir, f"{action}_oof_metrics.xlsx")
                pd.DataFrame([pre_row, post_row]).to_excel(met_path, index=False)
                logger.info(f"[Metrics] {met_path}")

                # consolidated row（oof）
                def _mk_comb(pre_dict, post_dict):
                    comb = {"Group": gname, "Action": action, "EvalSet": "oof",
                            "Tau": res.get("mean_tau", np.nan),
                            "post_method": res.get("best_method", "none"),
                            "N": post_dict.get("N", np.nan),
                            "Prevalence": post_dict.get("Prevalence", np.nan)}
                    for k in ["Brier","Brier_ref","BSS","LogLoss","ECE","MCE","Reliability",
                              "Resolution","Uncertainty","ROC_AUC","PR_AUC","SharpnessVar","SharpnessSD","n_bins"]:
                        comb[f"pre_{k}"]  = pre_dict.get(k, np.nan)
                        comb[f"post_{k}"] = post_dict.get(k, np.nan)
                    for k in ["Brier","LogLoss","ECE","ROC_AUC","PR_AUC"]:
                        try: comb[f"delta_{k}"] = float(comb[f"post_{k}"]) - float(comb[f"pre_{k}"])
                        except Exception: comb[f"delta_{k}"] = np.nan
                    return comb
                all_pred_rows.append(_mk_comb(res["oof_pre_metrics"], res["oof_post_metrics"]))

                # 校準法彙總表
                for r in res["calib_summary_rows"]:
                    all_calib_rows.append({"Group": gname, "Action": action, "eval_kind": "nested_inner_cv_summary", **r})

                # ---------- FINAL model（全 REAL+SYN；SYN down-weight） ----------
                train_df = pd.concat([d_real, d_syn], ignore_index=True)
                is_syn_tr = train_df["is_syn"].to_numpy(bool)
                w_syn = _syn_weight((~is_syn_tr).sum(), is_syn_tr.sum())
                W_tr = np.where(is_syn_tr, w_syn, 1.0)

                budget_final = _mcmc_budget(strategy="holdout", n_real=len(d_real))
                logger.info(f"[MCMC-Budget] {gname}:{action} FINAL "
                      f"chains={budget_final['num_chains']} warmup={budget_final['num_warmup']} samples={budget_final['num_samples']}")

                final_model = BayesianBetaRegressionModel(action, gname)
                final_model.fit(train_df["TP_mean"].values, train_df["CP_mean"].values, train_df["SP_mean"].values,
                                train_df[action].values, W=W_tr, **budget_final)

                # Final calibrator：用全部 REAL 擬合（與 nested 選出的 best 一致）
                p_real_all = final_model.predict_proba_mu(d_real["TP_mean"], d_real["CP_mean"], d_real["SP_mean"])
                y_real_all = (d_real[action].values > 0.5).astype(int)
                tau_grid = CFG.RL_TAU_GRID if action=="RL" else CFG.TAU_GRID
                plattC   = CFG.RL_PLATT_C  if action=="RL" else 1.0
                best_name = res.get("best_method", "none")
                if best_name == "temp":      cal = Calibrators.temp(p_real_all, y_real_all, tau_grid=tau_grid)
                elif best_name == "platt":   cal = Calibrators.platt(p_real_all, y_real_all, C=plattC)
                elif best_name == "isotonic":cal = Calibrators.isotonic(p_real_all, y_real_all)
                elif best_name == "beta":    cal = Calibrators.beta(p_real_all, y_real_all)
                else:                         cal = {"method": "none", "params": {}}
                final_model.set_calibrator(cal.get("method", "none"), cal.get("params", {}))

                if CFG.PARAM_EVAL_EARLY:
                    pe = run_param_eval_and_save(model=final_model, action=action, group=gname, out_dir=adir,
                                                 hdi_prob=CFG.HDI, rope=CFG.ROPE_WIDTH)
                    if pe is not None:
                        all_mcmc_rows.append(pe["mcmc_summary"])
                        all_pq_rows.append(pe["practical_significance"])
                        all_pp_rows.append(pe["prior_vs_posterior"])
                        all_corr_rows.append(pe["posterior_corr"])

                pkl_path = os.path.join(models_dir, f"{gname}_{action}.pkl")
                if final_model.calibrator is None:
                    final_model.set_calibrator("none", {})
                final_model.save_model(pkl_path)
                logger.info(f"[Model] Saved: {pkl_path}")

            else:
                # ---------- HOLDOUT：real-only 60/20/20 ----------
                n = len(d_real); idx = np.arange(n)
                if CFG.SPLIT_SHUFFLE:
                    rng = np.random.default_rng(CFG.SPLIT_SEED); rng.shuffle(idx)
                n_train = int(np.floor(n * CFG.TRAIN_FRAC))
                n_val   = int(np.floor(n * CFG.VAL_FRAC))
                train_idx = idx[:n_train]; val_idx = idx[n_train:n_train+n_val]; test_idx = idx[n_train+n_val:]

                train_real = d_real.iloc[train_idx].reset_index(drop=True)
                val_real   = d_real.iloc[val_idx].reset_index(drop=True)
                test_real  = d_real.iloc[test_idx].reset_index(drop=True)

                train_df = pd.concat([train_real, d_syn], ignore_index=True)
                is_syn_tr = train_df["is_syn"].to_numpy(bool)
                w_syn = _syn_weight((~is_syn_tr).sum(), is_syn_tr.sum())
                W_tr = np.where(is_syn_tr, w_syn, 1.0)

                budget = _mcmc_budget(strategy="holdout", n_real=len(train_real))
                logger.info(f"[MCMC-Budget] {gname}:{action} HOLDOUT "
                      f"chains={budget['num_chains']} warmup={budget['num_warmup']} samples={budget['num_samples']}")

                model = BayesianBetaRegressionModel(action, gname)
                model.fit(train_df["TP_mean"].values, train_df["CP_mean"].values, train_df["SP_mean"].values,
                          train_df[action].values, W=W_tr, **budget)

                if CFG.PARAM_EVAL_EARLY:
                    pe = run_param_eval_and_save(model=model, action=action, group=gname, out_dir=adir,
                                                 hdi_prob=CFG.HDI, rope=CFG.ROPE_WIDTH)
                    if pe is not None:
                        all_mcmc_rows.append(pe["mcmc_summary"])
                        all_pq_rows.append(pe["practical_significance"])
                        all_pp_rows.append(pe["prior_vs_posterior"])
                        all_corr_rows.append(pe["posterior_corr"])

                p_tr_pre_real = model.predict_proba_mu(train_real["TP_mean"], train_real["CP_mean"], train_real["SP_mean"])
                p_va_pre = model.predict_proba_mu(val_real["TP_mean"], val_real["CP_mean"], val_real["SP_mean"])
                p_te_pre = model.predict_proba_mu(test_real["TP_mean"], test_real["CP_mean"], test_real["SP_mean"])

                Y_tr_bin_real = (train_real[action].values > 0.5).astype(int)
                Y_va_bin = (val_real[action].values > 0.5).astype(int)
                Y_te_bin = (test_real[action].values > 0.5).astype(int)

                pre_val_metrics = _compute_metrics(Y_va_bin, p_va_pre, n_bins=N_BINS_THIS)
                pre_val_metrics.update({"Action": action, "Group": gname, "Tau": np.nan, "Stage": "pre", "EvalSet": "val", "Calib": "none"})
                pre_test_metrics = _compute_metrics(Y_te_bin, p_te_pre, n_bins=N_BINS_THIS)
                pre_test_metrics.update({"Action": action, "Group": gname, "Tau": np.nan, "Stage": "pre", "EvalSet": "test", "Calib": "none"})

                # 以 train_real 的 CV 挑校準法
                tau_grid = CFG.RL_TAU_GRID if action=="RL" else CFG.TAU_GRID
                plattC   = CFG.RL_PLATT_C  if action=="RL" else 1.0

                cv_rows_train = Calibrators.evaluate_all_cv(
                    p_tr_pre_real, Y_tr_bin_real, n_bins=N_BINS_THIS, folds=CV_FOLDS_THIS, methods=CAL_METHODS,
                    tau_grid=tau_grid, platt_C=plattC
                )
                for r in cv_rows_train:
                    all_calib_rows.append({"Group": gname, "Action": action, "eval_kind": "cv_train_real", **r})
                best_name = _pick_best_method_cv(cv_rows_train, is_rl=(action=="RL"))

                # 在 VALID 擬合選到的校準器
                if best_name == "temp":
                    fit_va = Calibrators.temp(p_va_pre, Y_va_bin, tau_grid=tau_grid)
                elif best_name == "platt":
                    fit_va = Calibrators.platt(p_va_pre, Y_va_bin, C=plattC)
                elif best_name == "isotonic":
                    fit_va = Calibrators.isotonic(p_va_pre, Y_va_bin)
                elif best_name == "beta":
                    fit_va = Calibrators.beta(p_va_pre, Y_va_bin)
                else:
                    fit_va = {"method": "none", "params": {}, "p_cal": _safe_clip(p_va_pre)}

                post_val_metrics = _compute_metrics(Y_va_bin, fit_va["p_cal"], n_bins=N_BINS_THIS)
                post_val = {**fit_va, **post_val_metrics, "Action": action, "Group": gname, "Stage": "post", "EvalSet": "val"}

                # Gate：VALID 沒改善就不校準
                gate_gain = CFG.RL_MIN_BRIER_GAIN if action=="RL" else CFG.MIN_BRIER_GAIN
                gate_ece  = CFG.RL_ALLOW_ECE_WORSEN if action=="RL" else CFG.ALLOW_ECE_WORSEN
                if not _should_apply_cal(pre_val_metrics, post_val,
                                         min_brier_gain=float(gate_gain),
                                         allow_ece_worsen=float(gate_ece)):

                    chosen_for_test = {"method": "none", "params": {}}
                    tau_val = np.nan
                else:
                    # 用 train_real + valid_real 重擬合校準器 → 測試
                    p_trva_pre = np.concatenate([p_tr_pre_real, p_va_pre])
                    Y_trva_bin = np.concatenate([Y_tr_bin_real, Y_va_bin])
                    if best_name == "temp":
                        fit_trva = Calibrators.temp(p_trva_pre, Y_trva_bin, tau_grid=tau_grid)
                    elif best_name == "platt":
                        fit_trva = Calibrators.platt(p_trva_pre, Y_trva_bin, C=plattC)
                    elif best_name == "isotonic":
                        fit_trva = Calibrators.isotonic(p_trva_pre, Y_trva_bin)
                    elif best_name == "beta":
                        fit_trva = Calibrators.beta(p_trva_pre, Y_trva_bin)
                    else:
                        fit_trva = {"method": "none", "params": {}}

                    chosen_for_test = {"method": fit_trva.get("method", "none"),
                                       "params": fit_trva.get("params", {})}
                    tau_val = post_val.get("params", {}).get("tau", np.nan)

                # prior shift（僅 RL）
                val_prev  = float(Y_va_bin.mean())
                test_prev = float(Y_te_bin.mean())

                model.set_calibrator(chosen_for_test.get("method", "none"), chosen_for_test.get("params", {}))
                p_te_post = model.calibrator.transform(p_te_pre) if model.calibrator else p_te_pre

                if (action == "RL") and _prior_shift_too_big(val_prev, test_prev, CFG.RL_PRIOR_SHIFT_THR):
                    pi_cal = float(np.mean(np.concatenate([Y_tr_bin_real, Y_va_bin])))
                    p_te_post = prior_shift_correct(p_te_post, pi_cal, test_prev)

                post_test_metrics = _compute_metrics(Y_te_bin, p_te_post, n_bins=N_BINS_THIS)
                post_test = {"method": model.calibrator.method if model.calibrator else "none",
                             "params": model.calibrator.params if model.calibrator else {},
                             "p_cal": p_te_post, **post_test_metrics,
                             "Action": action, "Group": gname, "Stage": "post", "EvalSet": "test"}

                met_path = os.path.join(adir, f"{action}_prob_metrics.xlsx")
                pd.DataFrame([pre_val_metrics, post_val, pre_test_metrics, post_test]).to_excel(met_path, index=False)
                logger.info(f"[Metrics] {met_path}")

                # consolidated rows
                def _mk_comb(pre_dict, post_dict, eval_set):
                    comb = {"Group": gname, "Action": action, "EvalSet": eval_set,
                            "Tau": tau_val if eval_set == "val" else np.nan,
                            "post_method": post_dict.get("method", "none"),
                            "N": post_dict.get("N", np.nan),
                            "Prevalence": post_dict.get("Prevalence", np.nan)}
                    for k in ["Brier","Brier_ref","BSS","LogLoss","ECE","MCE","Reliability",
                              "Resolution","Uncertainty","ROC_AUC","PR_AUC","SharpnessVar","SharpnessSD","n_bins"]:
                        comb[f"pre_{k}"]  = pre_dict.get(k, np.nan)
                        comb[f"post_{k}"] = post_dict.get(k, np.nan)
                    for k in ["Brier","LogLoss","ECE","ROC_AUC","PR_AUC"]:
                        try: comb[f"delta_{k}"] = float(comb[f"post_{k}"]) - float(comb[f"pre_{k}"])
                        except Exception: comb[f"delta_{k}"] = np.nan
                    return comb
                all_pred_rows.append(_mk_comb(pre_val_metrics,  post_val,  eval_set="val"))
                all_pred_rows.append(_mk_comb(pre_test_metrics, post_test, eval_set="test"))

                pkl_path = os.path.join(models_dir, f"{gname}_{action}.pkl")
                if model.calibrator is None:
                    model.set_calibrator("none", {})
                model.save_model(pkl_path)
                logger.info(f"[Model] Saved: {pkl_path}")

    # ===== Consolidation =====
    pred_df  = pd.DataFrame(all_pred_rows) if all_pred_rows else pd.DataFrame()
    calib_df = pd.DataFrame(all_calib_rows) if all_calib_rows else pd.DataFrame()
    mcmc_df  = pd.concat(all_mcmc_rows, ignore_index=True) if all_mcmc_rows else pd.DataFrame()
    pq_df    = pd.concat(all_pq_rows,   ignore_index=True) if all_pq_rows   else pd.DataFrame()
    pp_df    = pd.concat(all_pp_rows,   ignore_index=True) if all_pp_rows   else pd.DataFrame()
    corr_df  = pd.concat(all_corr_rows, ignore_index=True) if all_corr_rows else pd.DataFrame()

    # ============== Best calibrator JSON/YAML（以「實際採用」為準）==============
    def _to_yaml(d, indent=0):
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append("  " * indent + f"{k}:")
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        lines.extend(_to_yaml({kk: vv}, indent + 1))
                    else:
                        if isinstance(vv, str):
                            lines.append("  " * (indent + 1) + f"{kk}: {vv}")
                        else:
                            lines.append("  " * (indent + 1) + f"{kk}: {json.dumps(vv)}")
            else:
                lines.append("  " * indent + f"{k}: {json.dumps(v)}")
        return lines

    best_map = {}
    if not pred_df.empty:
        tmp = pred_df.copy()
        tmp["EvalSet"] = tmp["EvalSet"].astype(str).str.lower()
        # 偏好順序：oof > test > val
        rank_map = {"oof": 0, "test": 1, "val": 2}
        tmp["__rank__"] = tmp["EvalSet"].map(rank_map).fillna(3)

        tmp = tmp.sort_values(["Group", "Action", "__rank__"])
        chosen = tmp.drop_duplicates(subset=["Group", "Action"], keep="first")

        for _, r in chosen.iterrows():
            g, a = r["Group"], r["Action"]
            m = (r.get("post_method") or "none")
            entry = {"method": str(m), "evalset": str(r["EvalSet"])}
            # 可選：把改善幅度放進去
            if "delta_Brier" in r: entry["delta_Brier"] = float(r["delta_Brier"]) if pd.notna(r["delta_Brier"]) else None
            if "delta_ECE"   in r: entry["delta_ECE"]   = float(r["delta_ECE"])   if pd.notna(r["delta_ECE"])   else None
            tau = r.get("Tau")
            if str(m) == "temp" and pd.notna(tau):
                entry["param_tau"] = float(tau)
            best_map.setdefault(g, {})[a] = entry
    else:
        # 後備（理論上用不到）：若 pred_df 為空才用 CV 來填
        if not calib_df.empty:
            tmp = calib_df.copy()
            if "cv_mean_Brier" in tmp.columns:
                tmp = tmp.sort_values(["Group","Action","cv_mean_Brier"], ascending=[True,True,True])
            else:
                tmp = tmp.sort_values(["Group","Action","Brier","ECE","LogLoss"], ascending=[True,True,True,True,True])
            best_rows_df = tmp.drop_duplicates(subset=["Group","Action"], keep="first").copy()
            for _, r in best_rows_df.iterrows():
                g, a, m = r["Group"], r["Action"], r["method"]
                best_map.setdefault(g, {})[a] = {"method": str(m), "evalset": "cv_fallback"}

    out_json = os.path.join(CFG.RESULTS_DIR, "best_calibrators.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best_map, f, ensure_ascii=False, indent=2)

    out_yaml = os.path.join(CFG.RESULTS_DIR, "best_calibrators.yaml")
    with open(out_yaml, "w", encoding="utf-8") as f:
        f.write("\n".join(_to_yaml(best_map)))
    logger.info(f"[BestCal] JSON: {out_json}")
    logger.info(f"[BestCal] YAML: {out_yaml}")

    # Consolidated Excel
    consolidated_path = os.path.join(CFG.RESULTS_DIR, "all_results_summary.xlsx")
    with pd.ExcelWriter(consolidated_path) as xls:
        pred_df.to_excel(xls, sheet_name="predictive", index=False)
        calib_df.to_excel(xls, sheet_name="calibrations", index=False)
        mcmc_df.to_excel(xls, sheet_name="mcmc_summary", index=False)
        pq_df.to_excel(xls, sheet_name="practical_significance", index=False)
        pp_df.to_excel(xls, sheet_name="prior_vs_posterior", index=False)
        corr_df.to_excel(xls, sheet_name="posterior_corr", index=False)
    logger.info(f"[Consolidated] {consolidated_path}")
    logger.info(f"\nDone. See outputs under: {CFG.RESULTS_DIR}")

if __name__ == "__main__":
    main()
