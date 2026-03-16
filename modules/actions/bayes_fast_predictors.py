# -*- coding: utf-8 -*-
"""
Fast Bayesian Predictor Loader for FLOODABM
============================================

This module provides fast, vectorized Bayesian predictors for agent decision-making.
It loads pre-trained Bayesian regression models and caches the posterior mean weights
as .npz files for efficient subsequent loads.

Key Features:
- Supports both original models (models/) and optimized models (models_optimized/)
- Robust unpickling with shims for missing dependencies (JAX, NumPyro, etc.)
- Vectorized prediction using NumPy for high performance
- Optional probability calibration (Platt scaling, isotonic regression)

Public API:
    build_fast_predictors(actions_root) -> (predictor_owner, predictor_renter)
    bayes_fast_predictor(actions_root)  -> (predictor_owner, predictor_renter)  # alias

Example:
    >>> from pathlib import Path
    >>> pred_owner, pred_renter = build_fast_predictors(Path("modules/actions/models_optimized"))
    >>> result = pred_owner(TP=tp_array, CP=cp_array, SP=sp_array)
    >>> print(result["FI"])  # Flood Insurance adoption probabilities
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Callable

import numpy as np

__all__ = ["build_fast_predictors", "bayes_fast_predictor"]

# =============================================================================
# Configuration Constants
# =============================================================================

FEATURES = ["TP", "CP", "SP"]  # Trust in Policymaker, Community Perception, Self-efficacy
ACTIONS = ["FI", "EH", "BP", "RL"]  # Flood Insurance, Elevation, Barrier Protection, Relocation


# =============================================================================
# Dataclass Backport Check
# =============================================================================
# Third-party 'dataclasses' backport causes dill deserialization failures.
# Must use built-in dataclasses module only.

try:
    import dataclasses as _dc
    _dc_path = (_dc.__file__ or "").replace("\\", "/")
    if "site-packages" in _dc_path:
        raise RuntimeError(
            "Third-party 'dataclasses' backport detected in site-packages.\n"
            "This causes pickle/dill deserialization errors.\n"
            "Please uninstall: python -m pip uninstall -y dataclasses"
        )
except RuntimeError:
    raise
except Exception:
    pass


# =============================================================================
# Module Shims for Unpickling
# =============================================================================
# Create stub modules so pickle can resolve references to training-time packages
# (JAX, NumPyro, Flax, bayesian_engine) without actually installing them.

def _ensure_package(fullname: str):
    """Register a fake package module with __path__ attribute."""
    parts = fullname.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []  # Mark as package
            sys.modules[name] = mod
    return sys.modules[fullname]


def _ensure_module(fullname: str, attrs: dict = None):
    """Register a fake module with optional attributes."""
    parent = fullname.rsplit(".", 1)[0] if "." in fullname else None
    if parent:
        _ensure_package(parent)
    if fullname not in sys.modules:
        sys.modules[fullname] = types.ModuleType(fullname)
    if attrs:
        for key, value in attrs.items():
            setattr(sys.modules[fullname], key, value)
    return sys.modules[fullname]


# --- JAX shims ---
try:
    import jax  # noqa: F401
except Exception:
    _ensure_package("jax")
    _ensure_package("jax._src")
    _ensure_module("jax.numpy", vars(np))
    _ensure_module("jax.random", {"PRNGKey": lambda *a, **k: None})
if "jaxlib" not in sys.modules:
    _ensure_module("jaxlib")

# Ensure jax._src.array exists and patch it regardless of origin
_ensure_module("jax._src.array", {
    "_reconstruct_array": lambda x, *args: x
})


# --- NumPyro shims ---
try:
    import numpyro  # noqa: F401
    # Shim for older pickles if hmc_util is missing in newer numpyro
    try:
        import numpyro.infer.hmc_util
    except ImportError:
        class _StubFn:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, *args, **kwargs): return None

        _ensure_module("numpyro.infer.hmc_util", {
            "euclidean_kinetic_energy": _StubFn
        })
        _ensure_module("numpyro.infer.initialization", {
            "init_to_uniform": _StubFn
        })
        _ensure_module("numpyro.infer.util", {
            "_partial_args_kwargs": lambda *a, **k: ((), {}),
            "potential_energy": _StubFn,
            "_substitute_default_key": _StubFn,
            "constrain_fn": _StubFn
        })
        _ensure_module("numpyro.handlers", {
            "substitute": _StubFn
        })
except Exception:
    class _StubMCMC: pass
    class _StubNUTS: pass
    class _StubPredictive: pass
    
    _ensure_package("numpyro")
    _ensure_package("numpyro.distributions")
    _ensure_package("numpyro.infer")
    _ensure_module("numpyro.infer.mcmc", {"MCMC": _StubMCMC})
    _ensure_module("numpyro.infer.hmc", {"NUTS": _StubNUTS})
    # Shim for older pickles (v0.12.1 vs v0.15.0 compatibility)
    class _StubFn:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return None

    _ensure_module("numpyro.infer.hmc_util", {
        "euclidean_kinetic_energy": _StubFn
    })  # This fixes the optimized_2 load error
    _ensure_module("numpyro.infer.initialization", {
        "init_to_uniform": _StubFn
    })
    _ensure_module("numpyro.infer.util", {
        "_partial_args_kwargs": lambda *a, **k: ((), {}),
        "potential_energy": _StubFn,
        "_substitute_default_key": _StubFn,
        "constrain_fn": _StubFn
    })
    _ensure_module("numpyro.handlers", {
        "substitute": _StubFn
    })
    # Add to parent for direct import
    sys.modules["numpyro.infer"].MCMC = _StubMCMC
    sys.modules["numpyro.infer"].NUTS = _StubNUTS
    sys.modules["numpyro.infer"].Predictive = _StubPredictive

# --- Flax shims ---
try:
    import flax  # noqa: F401
except Exception:
    _ensure_package("flax")
    _ensure_module("flax.linen")

# --- bayesian_engine shims (from model training environment) ---
try:
    import bayesian_engine  # noqa: F401
except Exception:
    class _FlexibleStub:
        """Flexible stub class that accepts any pickle state."""
        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
        def __getattr__(self, name):
            if name == "posterior_samples":
                return {}
            return _FlexibleStub  # Return stub for nested class access
        def __call__(self, *args, **kwargs):
            return _FlexibleStub(*args, **kwargs)
    
    _FlexibleStub.action_beta_regression_model = _FlexibleStub
    
    _ensure_package("bayesian_engine")
    _ensure_module("bayesian_engine.model", {"BayesianBetaRegressionModel": _FlexibleStub})
    _ensure_module("bayesian_engine.config")
    _ensure_module("bayesian_engine.strategy")
    _ensure_module("bayesian_engine.calibration", {"Calibrator": _FlexibleStub})


# =============================================================================
# Deserialization Setup
# =============================================================================

try:
    import dill
    _unpickle_func = dill.load
except ModuleNotFoundError:
    import pickle
    _unpickle_func = pickle.load


# =============================================================================
# Math Utilities
# =============================================================================

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    z = np.asarray(z, dtype=np.float32)
    out = np.empty_like(z)
    pos_mask = z >= 0
    out[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    neg_mask = ~pos_mask
    exp_z = np.exp(z[neg_mask])
    out[neg_mask] = exp_z / (1.0 + exp_z)
    return out


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Logit function with numerical stability."""
    p = np.clip(np.asarray(p, dtype=np.float32), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


# =============================================================================
# Calibration Functions
# =============================================================================

def _identity_calibrator(p: np.ndarray) -> np.ndarray:
    """No calibration - returns input unchanged."""
    return p


def _make_platt_calibrator(A: float, B: float) -> Callable:
    """Create Platt scaling calibrator: sigmoid(A * logit(p) + B)."""
    A, B = float(A), float(B)
    return lambda p: _sigmoid(A * _logit(p) + B)


def _make_beta_calibrator(a: float, b: float, c: float) -> Callable:
    """Create beta calibrator (Kull et al., 2017).

    Formula: p_cal = sigmoid(a * log(p) + b * log(1-p) + c)

    Parameters a, b, c are fitted by minimizing log-loss during calibration.
    """
    a, b, c = float(a), float(b), float(c)
    _EPS = 1e-6  # Must be >= float32 epsilon (~1.2e-7) to prevent log(0)
    def _cal(p: np.ndarray) -> np.ndarray:
        p_clip = np.clip(np.asarray(p, dtype=np.float32), _EPS, 1.0 - _EPS)
        lp = np.log(p_clip)
        lq = np.log(1.0 - p_clip)
        return _sigmoid(a * lp + b * lq + c)
    return _cal


def _make_temp_calibrator(tau: float) -> Callable:
    """Create temperature scaling calibrator: sigmoid(logit(p) / tau)."""
    tau = float(tau)
    if tau <= 0:
        import warnings
        warnings.warn(f"Temperature tau={tau} is non-positive; defaulting to 1.0 (no scaling).")
        tau = 1.0
    return lambda p: _sigmoid(_logit(p) / tau)


def _make_isotonic_calibrator(x: np.ndarray, y: np.ndarray) -> Callable:
    """Create isotonic regression calibrator using linear interpolation."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return lambda p: np.interp(np.asarray(p, dtype=np.float32), x, y)


def _load_calibrator_from_npz(npz_path: Path) -> Callable | None:
    """Load a calibrator function from an .npz file's embedded parameters.

    Returns None if the npz file doesn't exist or has no calibrator data.
    """
    if not npz_path.exists():
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"[warn] Could not load calibrator from {npz_path.name}: {e}", file=sys.stderr)
        return None

    if "calibrator_method" not in data:
        return None

    method = str(data["calibrator_method"]).strip().lower()

    if method in ("beta",):
        # Kull et al. (2017) beta calibration: sigmoid(a*log(p) + b*log(1-p) + c)
        if not all(k in data for k in ("calibrator_param_a", "calibrator_param_b", "calibrator_param_c")):
            print(f"[warn] Incomplete beta calibrator params in {npz_path.name}", file=sys.stderr)
            return None
        a = float(data["calibrator_param_a"])
        b = float(data["calibrator_param_b"])
        c = float(data["calibrator_param_c"])
        return _make_beta_calibrator(a, b, c)

    if method in ("platt", "platt_scaling", "logistic"):
        if "calibrator_param_coef" not in data or "calibrator_param_intercept" not in data:
            print(f"[warn] Incomplete platt calibrator params in {npz_path.name}", file=sys.stderr)
            return None
        coef = float(data["calibrator_param_coef"])
        intercept = float(data["calibrator_param_intercept"])
        return _make_platt_calibrator(coef, intercept)

    if method in ("temp", "temperature", "temperature_scaling"):
        if "calibrator_param_tau" not in data:
            print(f"[warn] Missing temp calibrator tau in {npz_path.name}", file=sys.stderr)
            return None
        tau = float(data["calibrator_param_tau"])
        return _make_temp_calibrator(tau)

    return None


def _load_calibrators(calib_path: Path | None, group_tag: str) -> dict:
    """Load calibration functions from JSON config file (legacy fallback).

    Note: The primary calibrator loading path is now _load_calibrator_from_npz(),
    called in _load_group_models(). This function serves as a fallback for
    JSON-based configs using flat keys like 'OWNER_FI'.
    """
    calibrators = {action: _identity_calibrator for action in ACTIONS}

    if calib_path is None or not calib_path.exists():
        return calibrators

    config = json.loads(calib_path.read_text(encoding="utf-8"))

    # Support both flat keys ("OWNER_FI") and nested keys ({"owner": {"FI": ...}})
    prefix = group_tag.upper() + "_"
    group_lower = group_tag.lower()

    for action in ACTIONS:
        entry = None
        # Try flat key first
        key = prefix + action
        if key in config:
            entry = config[key]
        # Try nested key
        elif group_lower in config and isinstance(config[group_lower], dict):
            if action in config[group_lower]:
                entry = config[group_lower][action]

        if entry is None:
            continue

        cal_type = str(entry.get("type", entry.get("method", "platt"))).lower()

        if cal_type in ("beta",):
            a = entry.get("A", entry.get("a", 0.0))
            b = entry.get("B", entry.get("b", 1.0))
            c = entry.get("C", entry.get("c", 0.0))
            calibrators[action] = _make_beta_calibrator(a, b, c)
        elif cal_type in ("platt", "platt_scaling", "logistic"):
            A = entry.get("A", entry.get("a", entry.get("coef", 1.0)))
            B = entry.get("B", entry.get("b", entry.get("intercept", 0.0)))
            calibrators[action] = _make_platt_calibrator(A, B)
        elif cal_type in ("temp", "temperature", "temperature_scaling"):
            tau = entry.get("tau", 1.0)
            calibrators[action] = _make_temp_calibrator(tau)
        elif cal_type in ("iso", "isotonic"):
            x = entry.get("x") or entry.get("bins") or []
            y = entry.get("y") or entry.get("values") or []
            if len(x) >= 2 and len(y) == len(x):
                calibrators[action] = _make_isotonic_calibrator(np.array(x), np.array(y))

    return calibrators


# =============================================================================
# Model Weight Extraction
# =============================================================================

def _extract_posterior_mean(obj) -> tuple[np.ndarray, float]:
    """Extract posterior mean weights and bias from a model object.
    
    Supports multiple formats:
    - NumPyro-style: obj.posterior_samples dict
    - Scikit-learn style: obj.coef_, obj.intercept_
    - Dict format: {"weights": [...], "bias": float}
    
    Returns:
        (weights, bias): weight array of shape (3,) for [TP, CP, SP], and bias float
    
    Raises:
        ValueError: If posterior_samples contains unreadable values (JAX version mismatch)
    """
    n_features = len(FEATURES)
    
    # NumPyro-style model
    if hasattr(obj, "posterior_samples"):
        samples = obj.posterior_samples
        
        def get_param(key):
            for variant in (key, key.lower(), key.upper()):
                if hasattr(samples, "__contains__") and variant in samples:
                    val = samples[variant]
                    # Check for broken JAX reconstruction (callable instead of array)
                    if callable(val) and not hasattr(val, "shape"):
                        raise ValueError(
                            f"Parameter '{variant}' is a function reference instead of data. "
                            "This indicates a JAX/NumPyro version mismatch. "
                            "Please run export_weights.py in the original training environment "
                            "to convert this model to .npz format."
                        )
                    return np.asarray(val)
            return None
        
        weights = []
        for feat in FEATURES:
            arr = get_param(f"beta_{feat}") or get_param(f"beta_{feat.lower()}")
            weights.append(float(arr.mean()) if arr is not None else 0.0)
        
        b0 = get_param("beta_0")
        bias = float(b0.mean()) if b0 is not None else 0.0
        return np.array(weights, dtype=np.float32), bias
    
    # Scikit-learn style
    if hasattr(obj, "coef_"):
        w = np.asarray(obj.coef_, dtype=np.float32).reshape(-1)[:n_features]
        b = float(np.asarray(getattr(obj, "intercept_", 0.0)))
        return w.astype(np.float32), b
    
    # Dict format
    if isinstance(obj, dict):
        w = np.asarray(obj.get("weights", [0] * n_features), dtype=np.float32).reshape(-1)
        b = float(obj.get("bias", 0.0))
        if w.size < n_features:
            w = np.pad(w, (0, n_features - w.size))
        elif w.size > n_features:
            w = w[:n_features]
        return w.astype(np.float32), b
    
    # Fallback: zero weights
    return np.zeros(n_features, dtype=np.float32), 0.0


# =============================================================================
# Pickle Loading with Fallbacks
# =============================================================================

def _unpickle(path: Path):
    """Load a pickled model with robust error handling.
    
    Handles special case where third-party dataclasses backport
    causes _DataclassParams.match_args deserialization errors.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # First attempt: standard unpickle
    with open(path, "rb") as f:
        try:
            return _unpickle_func(f)
        except AttributeError as e:
            import traceback
            traceback.print_exc()  # Print full error to see what's missing
            error_msg = str(e)
            needs_fallback = "_DataclassParams" in error_msg or "match_args" in error_msg
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to unpickle {path.name}: {e}") from e
    
    if not needs_fallback:
        raise RuntimeError(f"Unpickle failed unexpectedly for {path.name}")
    
    # Fallback: custom unpickler for dataclasses compatibility
    try:
        import dill as _dill_module
    except ImportError:
        import pickle as _dill_module
    
    class _CompatDataclassParams:
        """Permissive stand-in for dataclasses._DataclassParams."""
        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class _FixedUnpickler(_dill_module.Unpickler):
        def find_class(self, module, name):
            if module == "dataclasses" and name == "_DataclassParams":
                return _CompatDataclassParams
            return super().find_class(module, name)
    
    with open(path, "rb") as f:
        try:
            return _FixedUnpickler(f).load()
        except Exception as e:
            raise RuntimeError(f"Fallback unpickle failed for {path.name}: {e}") from e


def _load_or_cache_weights(pkl_path: Path, npz_path: Path) -> tuple[np.ndarray, float]:
    """Load weights from cache or extract from pickle and cache.
    
    Priority: 
        1. Load from npz cache (FLOODABM format: w, b)
        1.5. Load from _fast sibling directory (training project format: posterior_xxx)
        2. Load from npz in same directory as pkl (training project format: posterior_xxx)
        3. Load from pkl and create npz cache
    
    Raises:
        RuntimeError: If pkl cannot be read or contains incompatible data format
    """
    # Priority 1: Check FLOODABM-style cache
    if npz_path.exists():
        data = np.load(npz_path)
        if "w" in data and "b" in data:
            return data["w"].astype(np.float32), float(data["b"])
    
    # Helper: extract calibrator keys from an npz for cache preservation
    def _cal_extras_from_npz(data) -> dict:
        """Extract calibrator_* keys from npz data for cache preservation."""
        extras = {}
        for k in data.keys():
            if k.startswith("calibrator_"):
                extras[k] = data[k]
        return extras

    # Helper function to extract weights from training project npz format
    def _extract_from_training_npz(npz_file: Path):
        """Returns (weights, bias, cal_extras) or None."""
        if not npz_file.exists():
            return None
        try:
            data = np.load(npz_file, allow_pickle=True)
            keys = list(data.keys())

            # Training project format uses 'posterior_xxx' keys
            if any(k.startswith("posterior_") for k in keys):
                weights = []
                for feat in FEATURES:
                    key = f"posterior_beta_{feat.lower()}"
                    if key in data:
                        arr = np.asarray(data[key])
                        weights.append(float(arr.mean()))
                    else:
                        weights.append(0.0)

                b0_key = "posterior_beta_0"
                bias = float(np.asarray(data[b0_key]).mean()) if b0_key in data else 0.0
                cal_extras = _cal_extras_from_npz(data)

                return np.array(weights, dtype=np.float32), bias, cal_extras
        except Exception:
            pass
        return None

    # Priority 1.5: Check _fast sibling directory for training project npz
    models_parent = pkl_path.parent.parent  # e.g., models/
    model_name = pkl_path.parent.name       # e.g., optimized
    fast_dir = models_parent / f"{model_name}_fast"
    fast_npz = fast_dir / pkl_path.with_suffix(".npz").name

    result = _extract_from_training_npz(fast_npz)
    if result is not None:
        weights, bias, cal_extras = result
        # Cache in FLOODABM format with calibrator params preserved
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, w=weights, b=np.float32(bias), **cal_extras)
        print(f"[cache] Converted training npz to fast cache: {npz_path.name}")
        return weights, bias

    # Priority 2: Check if training project generated npz alongside pkl
    same_dir_npz = pkl_path.with_suffix(".npz")
    result = _extract_from_training_npz(same_dir_npz)
    if result is not None:
        weights, bias, cal_extras = result
        # Cache in FLOODABM format with calibrator params preserved
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, w=weights, b=np.float32(bias), **cal_extras)
        print(f"[cache] Converted training npz to fast cache: {npz_path.name}")
        return weights, bias

    
    # Priority 3: Load from pkl and create cache
    try:
        obj = _unpickle(pkl_path)
        weights, bias = _extract_posterior_mean(obj)
    except ValueError as e:
        # JAX/NumPyro version mismatch - cannot extract data
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"MODEL CONVERSION REQUIRED: {pkl_path.name}\n"
            f"{'='*60}\n"
            f"This model was created with a different JAX/NumPyro version.\n"
            f"The system cannot automatically convert it.\n\n"
            f"SOLUTION:\n"
            f"1. Open the ORIGINAL training environment\n"
            f"2. Run: python export_weights.py\n"
            f"3. Copy the generated .npz files to: {npz_path.parent}\n"
            f"{'='*60}"
        ) from e
    
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, w=weights.astype(np.float32), b=np.float32(bias))
    print(f"[cache] Created fast cache: {npz_path.name}")
    
    return weights, bias


# =============================================================================
# Model Assembly
# =============================================================================

def _load_group_models(models_root: Path, group_tag: str, calib_path: Path | None):
    """Load all action models for a demographic group (owner or renter).

    Args:
        models_root: Path to directory containing .pkl model files
        group_tag: "owner" (homeowner) or "renter"
        calib_path: Optional path to calibrators JSON
        
    Returns:
        (W, b, calibrators): Weight matrix (4x3), bias vector (4,), calibrator list
    """
    # Detect model directory structure
    if (models_root / "owner_FI.pkl").exists() or (models_root / "renter_FI.pkl").exists():
        # Direct model directory (e.g., models_optimized/)
        models_dir = models_root
        cache_dir = Path(str(models_root) + "_fast")
    else:
        # Traditional structure with models/ subdirectory
        models_dir = models_root / "models"
        cache_dir = models_root / "models_fast"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize weight matrix and bias vector
    W = np.zeros((len(ACTIONS), len(FEATURES)), dtype=np.float32)
    b = np.zeros(len(ACTIONS), dtype=np.float32)
    
    # Load calibrators from JSON as fallback
    calibrator_map = _load_calibrators(calib_path, group_tag)
    calibrators = [_identity_calibrator] * len(ACTIONS)

    # Load each action model
    for j, action in enumerate(ACTIONS):
        pkl_file = models_dir / f"{group_tag}_{action}.pkl"
        if not pkl_file.exists():
            continue  # Missing model -> use zero weights

        npz_file = cache_dir / f"{group_tag}_{action}.npz"
        W[j, :], b[j] = _load_or_cache_weights(pkl_file, npz_file)

        # Priority 1: load calibrator from npz (cache or training)
        cal = _load_calibrator_from_npz(npz_file)
        if cal is None:
            # Try training npz in models_dir
            cal = _load_calibrator_from_npz(models_dir / f"{group_tag}_{action}.npz")
        if cal is not None:
            calibrators[j] = cal
        else:
            # Priority 2: JSON-based calibrator
            calibrators[j] = calibrator_map.get(action, _identity_calibrator)

    return W, b, calibrators


# =============================================================================
# Public API
# =============================================================================

def build_fast_predictors(
    actions_root: Path,
    calibrators_json: Path | None = None
) -> tuple[Callable, Callable]:
    """Build vectorized Bayesian predictors for both tenure groups.

    Args:
        actions_root: Path to model directory. Can be:
            - "modules/actions" (uses models/ subdirectory)
            - "modules/actions/models_optimized" (uses that directory directly)
        calibrators_json: Optional path to calibration config JSON

    Returns:
        (predictor_owner, predictor_renter): Two predictor functions

        Each predictor takes keyword arguments TP, CP, SP (arrays of equal length)
        and returns a dict mapping action names to probability arrays.

    Example:
        >>> pred_owner, pred_renter = build_fast_predictors(Path("modules/actions"))
        >>> probs = pred_owner(TP=np.array([0.5]), CP=np.array([0.5]), SP=np.array([0.5]))
        >>> print(probs["FI"])  # [0.423...]
    """
    actions_root = Path(actions_root)
    
    # Find calibrators file
    calib_path = None
    candidates = [
        calibrators_json,
        actions_root / "best_calibrators.json",
        actions_root / "config" / "best_calibrators.json",
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            calib_path = Path(cand)
            break
    
    # Load models for both groups
    W_owner, b_owner, cal_owner = _load_group_models(actions_root, "owner", calib_path)
    W_renter, b_renter, cal_renter = _load_group_models(actions_root, "renter", calib_path)
    
    def _create_predictor(W: np.ndarray, b: np.ndarray, calibrators: list) -> Callable:
        """Create a predictor closure with captured weights and calibrators."""
        W = np.asarray(W, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        
        def predict(*, TP, CP, SP) -> dict[str, np.ndarray]:
            """Predict action probabilities for given psychological features.
            
            Args:
                TP: Trust in Policymaker scores (N,)
                CP: Community Perception scores (N,)
                SP: Self-Efficacy perception scores (N,)
                
            Returns:
                Dict mapping action names (FI, EH, BP, RL) to probability arrays (N,)
            """
            X = np.stack([TP, CP, SP], axis=1).astype(np.float32)  # (N, 3)
            Z = X @ W.T + b  # Linear combination: (N, 4)
            P = _sigmoid(Z)  # Convert to probabilities
            
            # Apply calibration
            for j in range(P.shape[1]):
                P[:, j] = calibrators[j](P[:, j])
            
            return {action: P[:, j] for j, action in enumerate(ACTIONS)}
        
        return predict
    
    predictor_owner = _create_predictor(W_owner, b_owner, cal_owner)
    predictor_renter = _create_predictor(W_renter, b_renter, cal_renter)

    return predictor_owner, predictor_renter


# Alias for backward compatibility
def bayes_fast_predictor(
    actions_root: Path,
    calibrators_json: Path | None = None
) -> tuple[Callable, Callable]:
    """Alias for build_fast_predictors (backward compatibility)."""
    return build_fast_predictors(actions_root, calibrators_json)
