import numpy as np
from dataclasses import dataclass, field
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize

from .config import CFG
from .utils import _safe_clip, _logit, _inv_logit
from .metrics import _compute_metrics

@dataclass
class Calibrator:
    """
    Data class for storing calibration method and parameters.
    """
    method: str = "none"
    params: dict = field(default_factory=dict)
    
    def transform(self, p: np.ndarray) -> np.ndarray:
        """
        Applies the calibration transformation to probabilities.
        """
        p = _safe_clip(np.asarray(p))
        if self.method == "none": return p
        z = _logit(p)
        if self.method == "temp":
            tau = float(self.params.get("tau", 1.0)); return _safe_clip(_inv_logit(z / tau))
        if self.method == "platt":
            a = float(self.params.get("coef", self.params.get("a", 1.0)))
            b = float(self.params.get("intercept", self.params.get("b", 0.0)))
            return _safe_clip(_inv_logit(a * z + b))
        if self.method == "beta":
            a = float(self.params.get("a", 1.0)); b = float(self.params.get("b", 1.0)); c = float(self.params.get("c", 0.0))
            lp = np.log(p); lq = np.log(1 - p); return _safe_clip(_inv_logit(a * lp + b * lq + c))
        if self.method == "isotonic":
            iso_x = np.asarray(self.params.get("iso_x", [])); iso_y = np.asarray(self.params.get("iso_y", []))
            if iso_x.size and iso_y.size and iso_x.size == iso_y.size:
                return _safe_clip(np.interp(p, np.clip(iso_x, 1e-12, 1-1e-12), np.clip(iso_y, 1e-12, 1-1e-12)))
            return p
        return p

class Calibrators:
    """
    Collection of calibration methods.
    """
    @staticmethod
    def none(p: np.ndarray, y: np.ndarray) -> dict:
        p = _safe_clip(p); return {"method": "none", "params": {}, "p_cal": p}

    @staticmethod
    def temp(p: np.ndarray, y: np.ndarray, tau_grid: np.ndarray | None = None) -> dict:
        """
        Temperature scaling calibration.
        """
        p = _safe_clip(p); y = np.asarray(y).astype(int).ravel()
        tau_grid = CFG.TAU_GRID if tau_grid is None else np.asarray(tau_grid, float)
        z = _logit(p)
        best_tau, best_brier = None, np.inf
        for tau in tau_grid:
            pc = _inv_logit(z / tau)
            b = brier_score_loss(y, pc)
            if b < best_brier:
                best_brier, best_tau = b, tau
        p_cal = _inv_logit(z / float(best_tau))
        return {"method": "temp", "params": {"tau": float(best_tau)}, "p_cal": _safe_clip(p_cal)}

    @staticmethod
    def platt(p: np.ndarray, y: np.ndarray, C: float = 1.0, max_iter: int = 1000) -> dict:
        """
        Platt scaling (Logistic Regression) calibration.
        """
        p = _safe_clip(p); y = np.asarray(y).astype(int).ravel()
        X = _logit(p).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", C=float(C), max_iter=int(max_iter))
        lr.fit(X, y)
        p_cal = lr.predict_proba(X)[:, 1]
        return {"method": "platt",
                "params": {"coef": float(lr.coef_.ravel()[0]), "intercept": float(lr.intercept_.ravel()[0]), "C": float(C)},
                "p_cal": _safe_clip(p_cal)}

    @staticmethod
    def isotonic(p: np.ndarray, y: np.ndarray) -> dict:
        """
        Isotonic Regression calibration.
        """
        p = _safe_clip(p); y = np.asarray(y).astype(int).ravel()
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        p_cal = iso.predict(p)
        params = {"iso_x": iso.X_thresholds_.astype(float).tolist(),
                  "iso_y": iso.y_thresholds_.astype(float).tolist()}
        return {"method": "isotonic", "params": params, "p_cal": _safe_clip(p_cal)}

    @staticmethod
    def beta(p: np.ndarray, y: np.ndarray) -> dict:
        """
        Beta calibration.
        """
        p = _safe_clip(p); y = np.asarray(y).astype(int).ravel()
        lp = np.log(p); lq = np.log(1 - p)
        def transform(theta):
            a, b, c = theta; z = a * lp + b * lq + c
            return _safe_clip(_inv_logit(z))
        def obj(theta):
            pc = transform(theta); return log_loss(y, pc, labels=[0, 1])
        theta0 = np.array([1.0, 1.0, 0.0], dtype=float)
        res = minimize(obj, theta0, method="L-BFGS-B"); theta = res.x
        p_cal = transform(theta)
        return {"method": "beta",
                "params": {"a": float(theta[0]), "b": float(theta[1]), "c": float(theta[2])},
                "p_cal": _safe_clip(p_cal)}

    @staticmethod
    def evaluate_all(p: np.ndarray, y: np.ndarray, n_bins: int = 10, 
                     methods: tuple = ("temp", "platt", "isotonic", "beta", "none"),
                     tau_grid: np.ndarray | None = None, platt_C: float = 1.0) -> list[dict]:
        """
        Evaluates multiple calibration methods.
        """
        p = _safe_clip(p); y = np.asarray(y).astype(int).ravel()
        out = []
        def _eval_one(tag, ret):
            m = _compute_metrics(y, ret["p_cal"], n_bins=n_bins)
            row = {"method": tag, **ret, **m}; return row
        for tag in methods:
            if tag == "none":      ret = Calibrators.none(p, y)
            elif tag == "temp":    ret = Calibrators.temp(p, y, tau_grid=tau_grid)
            elif tag == "platt":   ret = Calibrators.platt(p, y, C=platt_C)
            elif tag == "isotonic":ret = Calibrators.isotonic(p, y)
            elif tag == "beta":    ret = Calibrators.beta(p, y)
            else: continue
            out.append(_eval_one(tag, ret))
        return out

    @staticmethod
    def evaluate_all_cv(p: np.ndarray, y: np.ndarray, n_bins: int = 10, folds: int = 5, 
                        methods: tuple = ("temp", "platt", "isotonic", "beta"),
                        tau_grid: np.ndarray | None = None, platt_C: float = 1.0) -> list[dict]:
        """
        Evaluates calibration methods using Cross-Validation.
        """
        p = _safe_clip(p); y = np.asarray(y).astype(int).ravel()
        kf = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=CFG.NP_SEED)
        results = []
        for tag in methods:
            fold_metrics, oof_pred = [], np.zeros_like(p, dtype=float)
            for tr_idx, te_idx in kf.split(np.zeros(len(y)), y):
                p_tr, y_tr = p[tr_idx], y[tr_idx]
                p_te, y_te = p[te_idx], y[te_idx]
                if tag == "temp":      fit = Calibrators.temp(p_tr, y_tr, tau_grid=tau_grid)
                elif tag == "platt":   fit = Calibrators.platt(p_tr, y_tr, C=platt_C)
                elif tag == "isotonic":fit = Calibrators.isotonic(p_tr, y_tr)
                elif tag == "beta":    fit = Calibrators.beta(p_tr, y_tr)
                else: continue
                if tag == "temp":
                    tau = fit["params"]["tau"]
                    p_te_cal = _inv_logit(_logit(_safe_clip(p_te)) / float(tau))
                elif tag == "platt":
                    a = fit["params"]["coef"]; b = fit["params"]["intercept"]
                    p_te_cal = _inv_logit(a * _logit(_safe_clip(p_te)) + b)
                elif tag == "isotonic":
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                    iso.fit(_safe_clip(p_tr), y_tr)
                    p_te_cal = iso.predict(_safe_clip(p_te))
                elif tag == "beta":
                    a = fit["params"]["a"]; b = fit["params"]["b"]; c = fit["params"]["c"]
                    lp = np.log(_safe_clip(p_te)); lq = np.log(1 - _safe_clip(p_te))
                    p_te_cal = _inv_logit(a * lp + b * lq + c)
                else:
                    p_te_cal = _safe_clip(p_te)
                p_te_cal = _safe_clip(p_te_cal)
                oof_pred[te_idx] = p_te_cal
                fold_metrics.append(_compute_metrics(y_te, p_te_cal, n_bins=n_bins))
            def _mean_std(key):
                vals = np.array([fm[key] for fm in fold_metrics], dtype=float)
                return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            mean_brier, std_brier = _mean_std("Brier")
            mean_logloss, std_logloss = _mean_std("LogLoss")
            mean_ece, std_ece = _mean_std("ECE")
            oof_m = _compute_metrics(y, oof_pred, n_bins=n_bins)
            results.append({
                "method": tag,
                "cv_mean_Brier": mean_brier, "cv_std_Brier": std_brier,
                "cv_mean_LogLoss": mean_logloss, "cv_std_LogLoss": std_logloss,
                "cv_mean_ECE": mean_ece, "cv_std_ECE": std_ece,
                "oof_Brier": oof_m["Brier"], "oof_LogLoss": oof_m["LogLoss"], "oof_ECE": oof_m["ECE"],
                "oof_ROC_AUC": oof_m["ROC_AUC"], "oof_PR_AUC": oof_m["PR_AUC"],
                "oof_SharpnessVar": oof_m["SharpnessVar"], "oof_SharpnessSD": oof_m["SharpnessSD"],
                "oof_Reliability": oof_m["Reliability"], "oof_Resolution": oof_m["Resolution"],
                "oof_Uncertainty": oof_m["Uncertainty"],
                "folds": int(folds),
            })
        return results

def _should_apply_cal(pre_val: dict, post_val: dict,
                      min_brier_gain: float = 1e-3,
                      allow_ece_worsen: float = 0.0) -> bool:
    """
    Gate with ONLY probability-quality criteria:
    - Brier must decrease at least min_brier_gain
    - ECE must not worsen more than allow_ece_worsen
    (AUC/PR are NOT used)
    """
    dbrier = float(post_val.get("Brier", np.inf)) - float(pre_val.get("Brier", np.inf))
    dece   = float(post_val.get("ECE",   np.inf)) - float(pre_val.get("ECE",   np.inf))
    ok_brier = (dbrier <= -min_brier_gain)
    ok_ece   = (dece <= allow_ece_worsen)
    return bool(ok_brier and ok_ece)
