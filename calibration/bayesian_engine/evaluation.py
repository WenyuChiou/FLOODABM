import os
import numpy as np
import pandas as pd
from jax import random
from numpyro.infer import Predictive
from .config import CFG
from .utils import setup_logger

logger = setup_logger(__name__)

class ParamEval:
    """
    Compact evaluator using ArviZ (if available).
    """
    def __init__(self, model):
        import arviz as az
        self.az = az
        if model.mcmc is None or model.posterior_samples is None:
            raise ValueError("Model needs to be fitted before parameter evaluation.")
        self.model = model
        try: self.idata = az.from_numpyro(model.mcmc, save_warmup=True)
        except Exception: self.idata = az.from_numpyro(model.mcmc)
        self.post = model.posterior_samples

    def _select_names(self, include: list[str] | None = None, exclude: list[str] | None = None) -> list[str]:
        """
        Selects parameter names for evaluation.
        """
        names = include if include else []
        if not names:
            for k, v in self.post.items():
                a = np.asarray(v)
                if a.ndim == 1 or (a.ndim == 2 and a.shape[1] == 1):
                    names.append(k)
        if exclude: names = [n for n in names if n not in set(exclude)]
        return names

    def mcmc_diagnostics(self, hdi_prob: float = CFG.HDI, include: list[str] | None = None, exclude: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
        """
        Computes MCMC diagnostics (r_hat, ess, etc.).
        """
        az = self.az
        names = self._select_names(include, exclude)
        summ = az.summary(self.idata, var_names=names, hdi_prob=hdi_prob)
        if "mcse_mean" in summ.columns and "sd" in summ.columns:
            summ["mcse_sd_ratio"] = (summ["mcse_mean"] / summ["sd"]).replace([np.inf, -np.inf], np.nan).clip(lower=0)
        else:
            summ["mcse_sd_ratio"] = np.nan
        run = {"divergences": np.nan, "BFMI": np.nan}
        try:
            extra = self.model.mcmc.get_extra_fields()
            if "diverging" in extra: run["divergences"] = int(np.sum(np.asarray(extra["diverging"])))
        except Exception: pass
        try:
            bfmi_arr = az.bfmi(self.idata)
            run["BFMI"] = float(np.asarray(bfmi_arr.to_array()).mean())
        except Exception: pass
        return summ, run

    def parameter_quality(self, rope: float = CFG.ROPE_WIDTH, hdi_prob: float = CFG.HDI, include: list[str] | None = None, exclude: tuple = ("phi",)) -> pd.DataFrame:
        """
        Computes parameter quality metrics (HDI, pd, ROPE).
        """
        names = self._select_names(include, exclude)
        rows = []
        a = (1 - hdi_prob) * 50
        lo_q, hi_q = a, 100 - a
        for k in names:
            x = np.asarray(self.post[k]).ravel()
            mean = float(np.mean(x)); sd = float(np.std(x, ddof=1))
            lo = float(np.percentile(x, lo_q)); hi = float(np.percentile(x, hi_q))
            pdv = float(max(np.mean(x>0), np.mean(x<0)))
            rope_pct = float(100.0 * np.mean((x>-rope) & (x<rope))) if (rope is not None) else np.nan
            rows.append({"param":k,"mean":mean,"sd":sd,f"HDI{int(hdi_prob*100)}_low":lo,f"HDI{int(hdi_prob*100)}_high":hi,"pd":pdv,"ROPE%":rope_pct})
        df = pd.DataFrame(rows)
        low = f"HDI{int(hdi_prob*100)}_low"; high = f"HDI{int(hdi_prob*100)}_high"
        df["sig90"] = np.where((df[low] > 0) | (df[high] < 0), "*", "")
        return df

    def prior_posterior_update(self, n_prior: int = 4000, include: list[str] | None = None, exclude: list[str] | None = None,
                               log_phi: bool = CFG.LOG_PHI_IN_PRIOR_POST) -> pd.DataFrame:
        """
        Computes prior-to-posterior update metrics (shrinkage, z-shift, KL).
        """
        pr = Predictive(self.model.action_beta_regression_model, num_samples=int(n_prior))(
            random.PRNGKey(CFG.JAX_MAIN_KEY),
            tp_mean=np.array([0.5]), cp_mean=np.array([0.5]), sp_mean=np.array([0.5])
        )
        names = self._select_names(include, exclude)
        rows = []
        for k in names:
            if k not in pr: continue
            prx = np.asarray(pr[k]).ravel()
            pox = np.asarray(self.post[k]).ravel()
            k_out = k
            if log_phi and k == "phi":
                prx = np.log(prx); pox = np.log(pox)
                k_out = "phi_log"
            m0, s0 = float(np.mean(prx)), float(np.std(prx, ddof=1))
            m1, s1 = float(np.mean(pox)), float(np.std(pox, ddof=1))
            shrink = np.nan if s0==0 else max(0.0, 1.0 - s1/s0)
            zshift = np.nan if s0==0 else (m1 - m0)/s0
            eps = 1e-12; s0=max(s0,eps); s1=max(s1,eps)
            kl10 = np.log(s0/s1) + (s1**2 + (m1-m0)**2)/(2*s0**2) - 0.5
            kl01 = np.log(s1/s0) + (s0**2 + (m1-m0)**2)/(2*s1**2) - 0.5
            jdiv = float(kl10 + kl01)
            rows.append({"param":k_out,"prior_mean":m0,"prior_sd":s0,"post_mean":m1,"post_sd":s1,"shrinkage":shrink,"z_shift":zshift,"J_div":jdiv})
        return pd.DataFrame(rows)

    def posterior_correlation(self, include: list[str] | None = None, exclude: list[str] | None = None) -> pd.DataFrame:
        """
        Computes posterior correlations between parameters.
        """
        names = self._select_names(include, exclude)
        if len(names) < 2:
            return pd.DataFrame(columns=["param_i","param_j","corr"])
        arr = np.column_stack([np.asarray(self.post[k]).ravel() for k in names])
        corr = np.corrcoef(arr, rowvar=False)
        rows = []
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                rows.append({"param_i": names[i], "param_j": names[j], "corr": float(corr[i,j])})
        return pd.DataFrame(rows)

    def report_text(self, rope: float = CFG.ROPE_WIDTH, hdi_prob: float = CFG.HDI, include: list[str] | None = None, exclude: list[str] | None = None) -> str:
        """
        Generates a text report of the parameter evaluation.
        """
        lines = ["="*68, f"Parameter Estimation Report: {self.model.action_name} [{self.model.group_name}]", "="*68]
        try:
            summ, run = self.mcmc_diagnostics(hdi_prob=hdi_prob, include=include, exclude=exclude)
            lines.append(f"\\n[MCMC] divergences={run.get('divergences',np.nan)}  BFMI={run.get('BFMI',np.nan):.3f}")
            bad_rhat = (summ.get("r_hat", pd.Series([1.0])).astype(float) > 1.01).any()
            low_ess = (summ.get("ess_bulk", pd.Series([np.inf])).astype(float) < 400).any() or \
                      (summ.get("ess_tail", pd.Series([np.inf])).astype(float) < 400).any()
            high_mcse = (summ.get("mcse_sd_ratio", pd.Series([0.0])).astype(float) > 0.10).any()
            if bad_rhat or low_ess or high_mcse or (run.get("divergences",0)>0):
                lines.append("  ⚠ Potential issues:")
                if bad_rhat: lines.append("    - r_hat > 1.01")
                if low_ess:  lines.append("    - ess_bulk/tail < 400")
                if high_mcse:lines.append("    - MCSE/SD > 0.10")
                if run.get("divergences",0)>0: lines.append("    - divergences > 0")
            else:
                lines.append("  ✓ Sampler diagnostics look good.")
        except Exception as e:
            lines.append(f"\\n[MCMC] unavailable: {e}")

        try:
            pq = self.parameter_quality(rope=rope, hdi_prob=hdi_prob, exclude=("phi",))
            lines.append("\\n[Practical significance] (pd / ROPE%)")
            for _, r in pq.iterrows():
                lines.append(
                    f"  {r['param']:<10s} mean={r['mean']:.3f} "
                    f"HDI{int(hdi_prob*100)}=[{r[f'HDI{int(hdi_prob*100)}_low']:.3f},"
                    f"{r[f'HDI{int(hdi_prob*100)}_high']:.3f}] {r['sig90']}  pd={r['pd']:.3f}" +
                    (f"  ROPE%={r['ROPE%']:.1f}" if np.isfinite(r['ROPE%']) else "")
                )
        except Exception as e:
            lines.append(f"\\n[Practical significance] unavailable: {e}")

        try:
            pp = self.prior_posterior_update(n_prior=4000)
            lines.append("\\n[Prior→Posterior] (shrinkage / z_shift / J_div)")
            for _, r in pp.iterrows():
                lines.append(f"  {r['param']:<10s} shrink={r['shrinkage']:.2f}  z_shift={r['z_shift']:.2f}  J_div={r['J_div']:.3f}")
        except Exception as e:
            lines.append(f"\\n[Prior→Posterior] unavailable: {e}")

        try:
            corr = self.posterior_correlation()
            max_abs = np.nan if corr.empty else np.max(np.abs(corr['corr']))
            lines.append(f"\\n[Posterior correlation] max |ρ|={max_abs if np.isfinite(max_abs) else np.nan:.3f}  (|ρ|>0.8 ⇒ collinearity risk)")
        except Exception as e:
            lines.append(f"\\n[Posterior correlation] unavailable: {e}")

        lines.append("\\nRefs: Gelman et al. (2020) Bayesian Workflow; Vehtari et al. (2021) r-hat/ESS; Makowski et al. (2019+) pd/ROPE.")
        return "\\n".join(lines)

def run_param_eval_and_save(model, action: str, group: str, out_dir: str, hdi_prob: float = CFG.HDI, rope: float = CFG.ROPE_WIDTH) -> dict | None:
    """
    Runs parameter evaluation and saves reports/Excel files.
    """
    os.makedirs(out_dir, exist_ok=True)
    base_model = getattr(model, "model", model)
    try: e = ParamEval(base_model)
    except Exception as ex:
        logger.warning(f"[ParamEval] Skipped: {ex}"); return None
    summ, run = e.mcmc_diagnostics(hdi_prob=hdi_prob)
    pq = e.parameter_quality(rope=rope, hdi_prob=hdi_prob, exclude=("phi",))
    pp = e.prior_posterior_update(n_prior=4000)
    corr = e.posterior_correlation()
    report = e.report_text(rope=rope, hdi_prob=hdi_prob)
    report_path = os.path.join(out_dir, f"{action}_{group}_param_inference_report.txt")
    with open(report_path, "w", encoding="utf-8") as f: f.write(report)
    excel_path = os.path.join(out_dir, f"{action}_{group}_param_inference.xlsx")
    with pd.ExcelWriter(excel_path) as xls:
        summ.to_excel(xls, sheet_name="mcmc_summary")
        pd.DataFrame([run]).to_excel(xls, sheet_name="run_diagnostics", index=False)
        pq.to_excel(xls, sheet_name="practical_significance", index=False)
        pp.to_excel(xls, sheet_name="prior_vs_posterior", index=False)
        corr.to_excel(xls, sheet_name="posterior_corr", index=False)
    logger.info(f"[ParamEval] Saved: {report_path}")
    logger.info(f"[ParamEval] Saved: {excel_path}")
    summ2 = summ.copy(); summ2.insert(0, "Group", group); summ2.insert(1, "Action", action); summ2["param"] = summ2.index
    pq2 = pq.copy(); pq2.insert(0, "Group", group); pq2.insert(1, "Action", action)
    pp2 = pp.copy(); pp2.insert(0, "Group", group); pp2.insert(1, "Action", action)
    corr2 = corr.copy(); corr2.insert(0, "Group", group); corr2.insert(1, "Action", action)
    return {"mcmc_summary": summ2.reset_index(drop=True),
            "practical_significance": pq2.reset_index(drop=True),
            "prior_vs_posterior": pp2.reset_index(drop=True),
            "posterior_corr": corr2.reset_index(drop=True)}
