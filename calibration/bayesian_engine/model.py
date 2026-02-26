import os
import sys
import dill
import numpy as np

# Check JAX installation before importing JAX modules
from .utils import check_jax_installation
_jax_ok, _jax_error = check_jax_installation()
if not _jax_ok:
    print(_jax_error, file=sys.stderr)
    sys.exit(1)

# Safe to import JAX now
import jax.numpy as jnp
import jax.nn as nn
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

from .config import CFG
from .utils import _safe_clip, setup_logger
from .calibration import Calibrator

logger = setup_logger(__name__)

class BayesianBetaRegressionModel:
    """
    Bayesian Beta Regression Model using NumPyro.
    
    Attributes:
        action_name (str): Name of the action being modeled.
        group_name (str | None): Name of the group (optional).
        auto_eval (bool): Whether to perform automatic evaluation.
        mcmc (MCMC | None): The fitted MCMC object.
        posterior_samples (dict | None): Samples from the posterior distribution.
        temp_tau (float | None): Temperature scaling parameter if applicable.
        calibrator (Calibrator | None): The calibrator object.
    """
    def __init__(self, action_name: str, group_name: str | None = None, auto_eval: bool = False):
        self.action_name = action_name
        self.group_name = group_name
        self.mcmc = None
        self.posterior_samples = None
        self.auto_eval = auto_eval
        self.temp_tau = None
        self.calibrator = None

    @staticmethod
    def action_beta_regression_model(tp_mean, cp_mean, sp_mean, y=None, w=None):
        # Informative priors for better sensitivity (optimized based on diagnostic analysis)
        # beta_0: intercept
        beta_0 = numpyro.sample("beta_0", dist.Normal(CFG.PRIOR_BETA_0_MEAN, CFG.PRIOR_BETA_0_SIGMA))
        
        # beta_tp: Trust in Policymaker - STRONG positive prior for sensitivity
        beta_tp = numpyro.sample("beta_tp", dist.Normal(CFG.PRIOR_BETA_TP_MEAN, CFG.PRIOR_BETA_TP_SIGMA))
        
        # beta_cp: Community Priority
        beta_cp = numpyro.sample("beta_cp", dist.Normal(CFG.PRIOR_BETA_CP_MEAN, CFG.PRIOR_BETA_CP_SIGMA))
        
        # beta_sp: Self Priority
        beta_sp = numpyro.sample("beta_sp", dist.Normal(CFG.PRIOR_BETA_SP_MEAN, CFG.PRIOR_BETA_SP_SIGMA))

        linear_predictor = beta_0 + beta_tp * tp_mean + beta_cp * cp_mean + beta_sp * sp_mean
        mu = numpyro.deterministic("mu", nn.sigmoid(linear_predictor))
        phi = numpyro.sample("phi", dist.Gamma(5.0, 1.0))

        alpha = mu * phi
        beta_param = (1 - mu) * phi

        if y is not None:
            eps = 1e-6
            y_adj = jnp.clip(y, eps, 1 - eps)
            lp = dist.Beta(alpha, beta_param).log_prob(y_adj)
            if w is None:
                numpyro.factor("ll", jnp.sum(lp))
            else:
                wv = jnp.asarray(w)
                numpyro.factor("ll_w", jnp.sum(wv * lp))
        else:
            numpyro.sample("y_pred", dist.Beta(alpha, beta_param))
        numpyro.deterministic("y_mean", mu)

    def fit(self, TP_mean: np.ndarray, CP_mean: np.ndarray, SP_mean: np.ndarray, Y: np.ndarray, 
            rng_key: jnp.ndarray | None = None,
            num_warmup: int | None = None, num_samples: int | None = None, 
            num_chains: int | None = None, chain_method: str | None = None,
            target_accept: float | None = None, dense_mass: bool | None = None, 
            W: np.ndarray | None = None):
        """
        Fits the Bayesian Beta Regression model using NUTS.

        Args:
            TP_mean: Array of TP mean values.
            CP_mean: Array of CP mean values.
            SP_mean: Array of SP mean values.
            Y: Target values (proportions).
            rng_key: JAX random key.
            num_warmup: Number of warmup steps.
            num_samples: Number of sampling steps.
            num_chains: Number of MCMC chains.
            chain_method: Chain execution method (e.g., 'vectorized').
            target_accept: Target acceptance probability for NUTS.
            dense_mass: Whether to use dense mass matrix.
            W: Optional weights for the likelihood.
        """
        
        if rng_key is None:
            rng_key = random.PRNGKey(CFG.JAX_MAIN_KEY)

        eps = 1e-6
        Y_adj = np.clip(Y, eps, 1 - eps)
        TP_mean = np.asarray(TP_mean); CP_mean = np.asarray(CP_mean); SP_mean = np.asarray(SP_mean)

        if CFG.MATCH_ORIGINAL:
            num_chains  = CFG.NUM_CHAINS_ORIG    if num_chains  is None else num_chains
            num_warmup  = CFG.WARMUP_ORIG        if num_warmup  is None else num_warmup
            num_samples = CFG.SAMPLES_ORIG       if num_samples is None else num_samples
            chain_method= CFG.CHAIN_METHOD_ORIG  if chain_method is None else chain_method
            target_accept = CFG.TARGET_ACCEPT    if target_accept is None else target_accept
            dense_mass = CFG.DENSE_MASS          if dense_mass is None else dense_mass
        else:
            num_chains  = CFG.NUM_CHAINS_ROBUST  if num_chains  is None else num_chains
            num_warmup  = CFG.WARMUP_ROBUST      if num_warmup  is None else num_warmup
            num_samples = CFG.SAMPLES_ROBUST     if num_samples is None else num_samples
            chain_method= CFG.CHAIN_METHOD_ROBUST if chain_method is None else chain_method
            target_accept = CFG.TARGET_ACCEPT    if target_accept is None else target_accept
            dense_mass = CFG.DENSE_MASS          if dense_mass is None else dense_mass

        logger.info(f"[Fit] N={len(Y_adj)} action={self.action_name} group={self.group_name}  "
                    f"chains={num_chains} warmup={num_warmup} draws={num_samples}")
        nuts_kernel = NUTS(self.action_beta_regression_model,
                           target_accept_prob=target_accept, dense_mass=dense_mass)
        self.mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,
                         num_chains=num_chains, chain_method=chain_method, progress_bar=True)
        self.mcmc.run(rng_key, tp_mean=TP_mean, cp_mean=CP_mean, sp_mean=SP_mean,
                      y=Y_adj, w=None if W is None else np.asarray(W))
        self.posterior_samples = self.mcmc.get_samples()
        try: self.mcmc.print_summary()
        except Exception: pass

    def predict_proba_mu(self, TP: np.ndarray, CP: np.ndarray, SP: np.ndarray, rng_key: jnp.ndarray | None = None) -> np.ndarray:
        """
        Predicts the mean probability (mu) from the posterior predictive distribution.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(CFG.JAX_MAIN_KEY + 1)
        if self.posterior_samples is None:
            raise ValueError("Model not fitted yet.")
        predictive = Predictive(self.action_beta_regression_model, self.posterior_samples)
        preds = predictive(rng_key, tp_mean=np.asarray(TP), cp_mean=np.asarray(CP), sp_mean=np.asarray(SP))
        p = np.asarray(preds["y_mean"]).mean(axis=0).ravel()
        return _safe_clip(p)

    def predict_proba(self, TP: np.ndarray, CP: np.ndarray, SP: np.ndarray, calibrated: bool = True,
                      rng_key: jnp.ndarray | None = None) -> np.ndarray:
        """
        Predicts probabilities, optionally applying calibration.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(CFG.JAX_MAIN_KEY + 1)
        p = self.predict_proba_mu(TP, CP, SP, rng_key=rng_key)
        if calibrated and (self.calibrator is not None):
            p = self.calibrator.transform(p)
        return _safe_clip(p)

    def set_calibrator(self, method: str = "none", params: dict | None = None):
        """
        Sets the calibrator for the model.
        """
        method = (method or "none").lower()
        params = dict(params or {})
        self.calibrator = Calibrator(method=method, params=params)
        self.temp_tau = float(params["tau"]) if (method == "temp" and "tau" in params) else None

    def save_model(self, filepath: str, also_save_npz: bool = True):
        """
        Saves the model to a file using dill, and optionally also as NPZ.
        
        Args:
            filepath: Path to save the .pkl file
            also_save_npz: If True, also saves a .npz version for cross-environment compatibility
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save PKL format
        with open(filepath, "wb") as f:
            dill.dump(self, f)
        logger.info(f"[Save PKL] {filepath}")
        
        # Also save NPZ format for cross-environment compatibility
        if also_save_npz and self.posterior_samples:
            npz_path = filepath.replace('.pkl', '.npz')
            self._save_as_npz(npz_path)
    
    def _save_as_npz(self, npz_path: str):
        """
        Save model weights as NPZ format for cross-environment compatibility.
        """
        flat_dict = {}
        
        # Metadata
        flat_dict['action_name'] = np.array(str(self.action_name or ''))
        flat_dict['group_name'] = np.array(str(self.group_name or ''))
        
        if self.temp_tau is not None:
            flat_dict['temp_tau'] = np.array(float(self.temp_tau))
        
        # Convert posterior samples from JAX to NumPy
        if self.posterior_samples:
            for key, value in self.posterior_samples.items():
                try:
                    flat_dict[f'posterior_{key}'] = np.asarray(value)
                except Exception as e:
                    logger.warning(f"Could not convert {key}: {e}")
        
        # Calibrator info
        if self.calibrator is not None:
            if hasattr(self.calibrator, 'method'):
                flat_dict['calibrator_method'] = np.array(str(self.calibrator.method))
            if hasattr(self.calibrator, 'params') and self.calibrator.params:
                for k, v in self.calibrator.params.items():
                    try:
                        flat_dict[f'calibrator_param_{k}'] = np.array(v)
                    except Exception:
                        pass
        
        np.savez(npz_path, **flat_dict)
        logger.info(f"[Save NPZ] {npz_path}")

    @staticmethod
    def load(filepath: str) -> "BayesianBetaRegressionModel":
        """
        Loads a model from a file.
        """
        with open(filepath, "rb") as f:
            obj = dill.load(f)
        if isinstance(obj, dict):
            m = BayesianBetaRegressionModel(obj.get("action_name"), obj.get("group_name"),
                                            obj.get("auto_eval", False))
            m.mcmc = obj.get("mcmc")
            m.posterior_samples = obj.get("posterior_samples")
            m.temp_tau = obj.get("temp_tau", None)
            m.calibrator = obj.get("calibrator", None)
            return m
        return obj
