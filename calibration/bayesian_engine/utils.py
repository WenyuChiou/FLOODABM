import numpy as np

import logging
import sys

def check_jax_installation():
    """
    Safely checks if JAX and NumPyro are properly installed.
    
    Returns:
        tuple: (success: bool, error_message: str | None)
    """
    try:
        import jax
        import jaxlib
        import numpyro
        return (True, None)
    except ImportError as e:
        error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          JAX INSTALLATION ERROR                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

❌ Failed to import JAX/NumPyro: {str(e)}

This error typically occurs when JAX/jaxlib is not installed or has DLL loading issues.

📋 WINDOWS INSTALLATION GUIDE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CPU-only installation (recommended):
   pip install --upgrade "jax[cpu]"
   pip install numpyro

2. If you have CUDA GPU (advanced):
   pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   pip install numpyro

3. Verify installation:
   python verify_jax.py

4. Common issues:
   - Missing Visual C++ Redistributable → Download from Microsoft
   - Conflicting package versions → Try creating fresh virtual environment
   - Path issues → Ensure Python and pip are in system PATH

📖 For detailed instructions, see: install_jax.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return (False, error_msg)
    except Exception as e:
        error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          JAX RUNTIME ERROR                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

❌ Unexpected error when loading JAX: {type(e).__name__}: {str(e)}

This may indicate a DLL loading failure or incompatible library versions.

🔧 TROUBLESHOOTING STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Uninstall and reinstall JAX:
   pip uninstall jax jaxlib numpyro -y
   pip install --upgrade "jax[cpu]"
   pip install numpyro

2. Check for conflicting packages:
   pip list | findstr jax

3. Try creating a fresh virtual environment

4. See install_jax.md for detailed troubleshooting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return (False, error_msg)

def setup_logger(name: str = "BayesianEngine", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with a standard format.
    
    Args:
        name: The name of the logger.
        level: The logging level (default: logging.INFO).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def _safe_clip(p, eps=1e-12):
    return np.clip(p, eps, 1 - eps)

def _logit(p):
    p = _safe_clip(p)
    return np.log(p / (1 - p))

def _inv_logit(z):
    return 1 / (1 + np.exp(-z))

def _adaptive_bins(N, lo=5, hi=10):
    N = int(max(1, N))
    return int(max(lo, min(hi, int(np.sqrt(N)))))

def _calib_methods_by_n(N):
    if N < 30:
        return ["temp", "platt"]
    if N < 60:
        return ["temp", "platt", "beta"]
    return ["temp", "platt", "isotonic", "beta"]

def _cv_folds_by_n(N):
    return 3 if N < 40 else 5
