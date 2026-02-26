import os
import numpy as np

class CFG:
    # Reproducibility
    NP_SEED = 0
    PY_SEED = 0
    JAX_MAIN_KEY = 0

    # Inference
    MATCH_ORIGINAL = True
    PARAM_EVAL_EARLY = True

    # "Original-like"
    NUM_CHAINS_ORIG = 4
    WARMUP_ORIG = 4000
    SAMPLES_ORIG = 1200
    CHAIN_METHOD_ORIG = "vectorized"
    TARGET_ACCEPT = 0.9
    DENSE_MASS = True

    # Evaluation mode
    RL_EVAL_MODE = "auto"   # "auto" / "nested" / "holdout"
    RL_KEEP_RL_METHODS = True  # ★ RL 只用 RL_CALIB_METHODS（見下）

    # Alternative preset (not used when MATCH_ORIGINAL=True)
    NUM_CHAINS_ROBUST = 4
    WARMUP_ROBUST = 4000
    SAMPLES_ROBUST = 1000
    CHAIN_METHOD_ROBUST = "vectorized"

    # Calibration & evaluation (global defaults; 小樣本會被自適應覆蓋)
    N_BINS = 10
    TAU_GRID = np.linspace(0.85, 1.25, 21)      # default for non-RL
    RL_TAU_GRID = np.linspace(0.80, 1.60, 33)   # ★ 更寬的 RL 專用溫度網格

    # --- dataset split (for n_real ≥ 60; real-only) ---
    TRAIN_FRAC = 0.60
    VAL_FRAC   = 0.20
    TEST_FRAC  = 0.20
    SPLIT_SHUFFLE = False
    SPLIT_SEED    = 42

    # --- calibration selection ---
    USE_TRAIN_BASED_CAL = True
    CALIB_CV_FOLDS = 5
    CALIB_METHODS = ["temp","platt","isotonic","beta","none"]

    # --- gate by VALID improvement ---
    # ★ 僅以機率品質為核心：Brier 改善＋ECE 不變/下降
    MIN_BRIER_GAIN = 1e-4
    ALLOW_ECE_WORSEN = 0.0

    # --- auto strategy thresholds ---
    SMALL_NESTED_THRESHOLD = 100    # n_real < 100 → RS-Nested CV (restored from user's previous setting)
    OUTER_FOLDS_SMALL = 3
    OUTER_REPEATS_SMALL = 20
    INNER_FOLDS_SMALL = 3

    # --- RL-specific knobs (ECE-oriented) ---
    RL_FORCE_NESTED = False
    RL_CALIB_METHODS = ["temp", "platt"]  # ★ 僅 temp/platt；gate 失敗時會回退 none
    RL_MIN_BRIER_GAIN = 5e-3              # VALID 至少降低 0.005 才啟用
    RL_ALLOW_ECE_WORSEN = 0.0             # ECE 不得上升（小樣本可放寬到 +1e-3）
    RL_STD_PENALTY = 0.25                 # 內層 CV 的穩定性懲罰係數
    RL_ECE_WEIGHT = 0.5                   # ★ 內層 CV 對 ECE 的權重
    RL_BRIER_TIE_EPS = 3e-3               # Brier 幾乎相同時的平手帶
    RL_PRIOR_SHIFT_THR = 0.20             # |π_val - π_test| 超過就對 test 做先驗平移
    RL_PLATT_C = 0.2                      # ★ Platt 正則（越小越強）

    # ROPE and HDI for parameter reporting
    ROPE_WIDTH = 0.10
    HDI = 0.90

    # Actions & groups
    ACTIONS = ["FI", "EH", "BP", "RL"]
    GROUPS = [
        ("owner_variable", "owner"),
        ("renter_variable", "renter"),
    ]

    # Paths
    # Note: os.getcwd() will be evaluated when this module is imported.
    # It is better to rely on absolute paths or relative to the script execution.
    # For now, we keep it as is but users should be aware of CWD.
    RESULTS_DIR = os.path.join(os.getcwd(), "bayesian_model_results_slim_plus")
    DATA_FILE = os.path.join(os.getcwd(), "data", "data.xlsx")

    LOG_PHI_IN_PRIOR_POST = False

    # --- Prior Hyperparameters (Sensitivity Tuning) ---
    PRIOR_BETA_0_MEAN = -1.0
    PRIOR_BETA_0_SIGMA = 2.0
    
    # Strong sensitivity: Mean 6.0 (approx slope ~1.5 at p=0.5 -> delta ~0.7), Sigma 0.5 (tight)
    PRIOR_BETA_TP_MEAN = 3.0
    PRIOR_BETA_TP_SIGMA = 0.5
    
    PRIOR_BETA_CP_MEAN = 1.0
    PRIOR_BETA_CP_SIGMA = 1.5
    
    PRIOR_BETA_SP_MEAN = 0.5
    PRIOR_BETA_SP_SIGMA = 1.5
