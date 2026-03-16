# config.py
# Centralized Configuration for Threat Perception Calibration (Owner/Renter)

import numpy as np
from pathlib import Path

# --- File Paths ---
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "cal_data.xlsx"
OUTDIR = Path(__file__).resolve().parent / "outputs"

# --- Column Mappings ---
COLS = dict(
    PA=["PA_mean", "pa", "place_attachment"],
    SC=["SC_mean", "sc", "social_capital"],
    TP=["TP_mean", "tp", "threat_perception"],
    T_LO=["t_lo", "time", "years_since", "q15_year", "tmin"],
    T_HI=["t_hi", "time_max", "tmax"]
)
COL_Q15_RAW = "q15"  # Excel column Q15 = original survey Q12 (flood experience timing)

# --- Core Logic Settings ---
IMPUTE_MEAN = 5.0
IMPUTE_STD = 1.0
RANDOM_SEED = 42

# --- Optimization Grid Settings (owner/renter) ---
GRID_SETTINGS = dict(
    owner=dict(
        alpha_range=(0.10, 0.50), alpha_steps=25,
        beta_range=(0.10, 0.50),  beta_steps=25,
        r_target=0.5,
        lambda_pen=0.40,
        bounds_tau0=(1.0, 12.0),
        bounds_d=(4.0, 50.0),
        bounds_k=(0.03, 0.6),
    ),
    renter=dict(
        alpha_range=(0.10, 0.50), alpha_steps=25,
        beta_range=(0.10, 0.50),  beta_steps=25,
        r_target=0.5,
        lambda_pen=0.30,
        bounds_tau0=(1.0, 12.0),
        bounds_d=(4.0, 50.0),
        bounds_k=(0.01, 0.6),
    ),
)

# --- Cross-Group Constraints ---
CROSS_CONSTRAINTS = dict(
    min_gap_alpha=0.1,
    min_gap_beta=0.1
)

# --- Computation Settings ---
COMPUTE = dict(
    n_quantiles=9,
    t_max=40.0,
    dt=0.05,
)
