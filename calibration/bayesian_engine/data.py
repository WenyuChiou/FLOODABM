import os
import numpy as np
import pandas as pd
from .config import CFG
# Assuming src is in the python path or relative import works if we are running from root
from src.util_2 import load_data

def _flag_synthetic(df, source_col="Source"):
    if source_col in df.columns:
        s = df[source_col].astype(str).str.strip().str.lower()
        return s.eq("synthetic").to_numpy(dtype=bool)
    return np.zeros(len(df), dtype=bool)

def _syn_weight(n_real, n_syn):
    if n_syn <= 0: return 0.0
    return float(min(1.0, n_real / n_syn))

def get_data():
    if not os.path.exists(CFG.DATA_FILE):
        raise FileNotFoundError(f"Cannot find data.xlsx at {CFG.DATA_FILE}")
    data = load_data(CFG.DATA_FILE, normalize_actions=True)
    if not data:
        raise RuntimeError("load_data returned empty result.")
    return data
