# core/model.py
# Core Mathematical Model: Saturated Tau Decay

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d

LN2 = np.log(2.0)

def tau_sat(t, tau0, tau_inf, k):
    """Saturated exponential: increasing from tau0 (t=0) to tau_inf as t->inf."""
    t = np.asarray(t, dtype=float)
    val = tau_inf - (tau_inf - tau0) * np.exp(-k * np.clip(t, 0.0, None))
    return val

def eff_closed_form(t, tau0, tau_inf, k):
    """Closed-form integral Eff(t) = int_0^t 1/tau(u) du."""
    t = np.asarray(t, dtype=float)
    tau_diff = tau_inf - tau0
    num = tau_inf - tau_diff * np.exp(-k * t)
    num = np.maximum(num, 1e-12)
    tau0_safe = max(tau0, 1e-12)
    term1 = t / tau_inf
    term2 = (1.0 / (k * tau_inf)) * np.log(num / tau0_safe)
    return term1 + term2

def precompute_eff_grid(t_max, dt, tau0, tau_inf, k):
    """Precomputes Eff(t) on a grid for fast interpolation."""
    t_grid = np.arange(0.0, t_max + dt, dt)
    eff_vals = eff_closed_form(t_grid, tau0, tau_inf, k)
    return t_grid, eff_vals

def eff_at_times(times, t_grid, eff_grid):
    return np.interp(times, t_grid, eff_grid)

def fit_tau_sat_given_ab(df, a, b, r_target, lambda_pen,
                         n_q=9, t_max=40.0, dt=0.05,
                         bounds_tau0=(1.0, 12.0),
                         bounds_d=(4.0, 50.0),
                         bounds_k=(0.01, 1.0)):
    """
    Given fixed alpha (a) and beta (b), find optimal tau parameters (tau0, delta, k).
    TP0 is profiled out analytically.
    """
    PA = df["PA"].values
    SC = df["SC"].values
    TP = df["TP"].values
    times = 0.5 * (df["t_lo"].values + df["t_hi"].values)
    w_eff = a * (1.0 - PA) + b * SC

    bounds = [bounds_tau0, bounds_d, bounds_k]

    def objective(x):
        t0, dlt, k_val = x
        t_inf = t0 + dlt
        Eff = eff_closed_form(times, t0, t_inf, k_val)
        g = np.exp(-LN2 * w_eff * Eff)
        num = np.sum(TP * g)
        den = np.sum(g * g) + 1e-12
        c_hat = num / den
        y_pred = c_hat * g
        mse = np.mean((TP - y_pred)**2)
        pen = 0.0
        eff5 = eff_closed_form(5.0, t0, t_inf, k_val)
        w_ref = a * 0.5 + b * 0.5
        pred_ratio_5 = np.exp(-LN2 * w_ref * eff5)
        pen += lambda_pen * (pred_ratio_5 - r_target)**2
        return mse + pen

    res = differential_evolution(objective, bounds, strategy='best1bin',
                                 popsize=10, mutation=(0.5, 1.0),
                                 recombination=0.7, seed=42)

    best_t0, best_delta, best_k = res.x
    best_t_inf = best_t0 + best_delta

    Eff = eff_closed_form(times, best_t0, best_t_inf, best_k)
    g = np.exp(-LN2 * w_eff * Eff)
    c_hat = np.sum(TP * g) / (np.sum(g * g) + 1e-12)
    y_pred = c_hat * g
    mse = np.mean((TP - y_pred)**2)
    mae = np.mean(np.abs(TP - y_pred))

    def get_ratio(yr):
        e = eff_closed_form(yr, best_t0, best_t_inf, best_k)
        w_mean = a * np.mean(PA) + b * np.mean(SC)
        return float(np.exp(-LN2 * w_mean * e))

    return dict(
        alpha=a, beta=b,
        tau0=best_t0, tau_inf=best_t_inf, k=best_k,
        TP0_hat=float(c_hat),
        RMSE=np.sqrt(mse), MAE=mae,
        PenalizedObj=res.fun,
        TP5_TP0=get_ratio(5.0),
        TP10_TP0=get_ratio(10.0),
        TP15_TP0=get_ratio(15.0)
    )
