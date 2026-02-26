import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
from .utils import _safe_clip

def _compute_metrics(y_true, p, n_bins=10):
    yb = np.asarray(y_true).astype(int).ravel()
    p = _safe_clip(np.asarray(p).ravel())
    N = len(p); pi = float(yb.mean())
    brier = float(brier_score_loss(yb, p))
    brier_ref = pi * (1 - pi)
    bss = np.nan if brier_ref == 0 else 1 - (brier / brier_ref)
    nll = float(log_loss(yb, p, labels=[0, 1]))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins) - 1
    ece = mce = 0.0
    reliability = resolution = 0.0
    for b in range(n_bins):
        m = (bin_ids == b)
        nb = int(m.sum())
        if nb == 0: continue
        w = nb / N
        p_true = float(yb[m].mean()); p_pred = float(p[m].mean())
        diff = abs(p_true - p_pred)
        ece += w * diff; mce = max(mce, diff)
        reliability += w * (p_pred - p_true) ** 2
        resolution += w * (p_true - pi) ** 2
    uncertainty = pi * (1 - pi)
    try: roc_auc = float(roc_auc_score(yb, p))
    except Exception: roc_auc = np.nan
    try: pr_auc = float(average_precision_score(yb, p))
    except Exception: pr_auc = np.nan
    sharp_var = float(np.mean(p * (1 - p))); sharp_sd = float(np.std(p))
    return {"N": int(N), "Prevalence": float(pi), "Brier": brier, "Brier_ref": float(brier_ref),
            "BSS": float(bss), "LogLoss": nll, "ECE": float(ece), "MCE": float(mce),
            "Reliability": float(reliability), "Resolution": float(resolution), "Uncertainty": float(uncertainty),
            "ROC_AUC": float(roc_auc), "PR_AUC": float(pr_auc),
            "SharpnessVar": sharp_var, "SharpnessSD": float(sharp_sd), "n_bins": int(n_bins)}
