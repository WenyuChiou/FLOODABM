# -*- coding: utf-8 -*-
"""
FLOODABM/utils/config_loader.py

.. WARNING:: LEGACY — NOT USED AT RUNTIME (verified 2026-06).
    The live runners hardcode ``CONFIG_PATH = config/abm_params.yaml`` (main.py)
    and load it via ``utils._helpers.load_yaml_cfg``. The ``--cfg`` / ``-c``
    flag and ``FLOODABM_CONFIG`` env var below are NOT honored by
    main.py / main_mc.py, and this loader looks for ``abm_params.json`` (not the
    shipped ``.yaml``). Kept for reference only.

Centralized config & path loading for FLOODABM runners.
- Supports external configs (CLI --cfg / env FLOODABM_CONFIG)
- Resolves relative files w.r.t. config folder (then modules/, then CWD)
- Provides helpers to load households (CSV/Excel) and normalize columns
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

def pick_cfg_path(project_root: Path, modules_root: Path) -> Path:
    # CLI: --cfg <path> or --cfg=<path>
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a in ("--cfg", "-c") and i + 1 < len(argv):
            p = Path(argv[i+1]);  return p if p.is_absolute() else (Path.cwd()/p)
        if a.startswith("--cfg="):
            p = Path(a.split("=",1)[1]);  return p if p.is_absolute() else (Path.cwd()/p)
    # ENV
    env_p = os.environ.get("FLOODABM_CONFIG")
    if env_p:
        p = Path(env_p); return p if p.is_absolute() else (Path.cwd()/p)
    # Defaults (prefer external config/, then internal actions/config/)
    cands = [
        project_root / "config" / "abm_params.json",
        modules_root / "actions" / "config" / "abm_params.json",
        Path.cwd() / "config" / "abm_params.json",
    ]
    for c in cands:
        if c.exists(): return c
    raise FileNotFoundError("Could not locate abm_params.json. Use --cfg or set FLOODABM_CONFIG.")

def _resolve(base_dir: Path, modules_root: Path, path_like: Optional[str]) -> Optional[Path]:
    if not path_like: return None
    p = Path(path_like)
    if p.is_absolute(): return p
    for root in (base_dir, modules_root, Path.cwd()):
        cand = root / p
        if cand.exists(): return cand.resolve()
    return (base_dir / p).resolve()

def load_abm_config(project_root: Path, modules_root: Path) -> Dict[str, Any]:
    cfg_path = pick_cfg_path(project_root, modules_root)
    cfg_dir  = cfg_path.parent
    cfg: Dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))

    files = cfg.get("files", {}) or {}
    paths = {
        "households":   _resolve(cfg_dir, modules_root, files.get("households")),
        "depths":       _resolve(cfg_dir, modules_root, files.get("depths_overall")),
        "mg_share":     _resolve(cfg_dir, modules_root, files.get("mg_share")),
        "policy":       _resolve(cfg_dir, modules_root, files.get("policy")),
    }

    out_root = _resolve(cfg_dir, modules_root, (cfg.get("paths", {}) or {}).get("outputs_root", "FLOODABM_outputs"))
    figs_dir   = out_root / "figs"
    states_dir = out_root / "states"
    finance_dir= out_root / "finance"
    for d in (out_root, figs_dir, states_dir, finance_dir):
        d.mkdir(parents=True, exist_ok=True)

    params = {
        "seed":        int(cfg.get("seed", 2011)),
        "years_step":  float(cfg.get("years_step", 1.0)),
        "years":       cfg.get("years", {}) or {},  # runner will use depths to determine default year range
        "tp_decay":    cfg.get("tp_decay_params", {}) or {},
        "tp_config":   cfg.get("tp_config", {}) or {},
        "flood":       cfg.get("flood", {}) or {},
        "dynamics":    cfg.get("action_dynamics", {}) or {},
    }

    return {
        "cfg": cfg,
        "cfg_path": cfg_path,
        "cfg_dir": cfg_dir,
        "paths": {"outputs_root": out_root, "figs": figs_dir, "states": states_dir, "finance": finance_dir, **paths},
        "params": params,
    }

# ---------- Households helpers ----------
_RENAME = {
    "TRACT_GEOID": "tract_geoid", "Tract": "tract_geoid", "tract": "tract_geoid",
    "GROUP": "group", "Group": "group",
    "RCV_kUSD": "rcv_kUSD", "CONTENTS_kUSD": "contents_kUSD", "contents": "contents_kUSD",
    "elev_ft": "elev_ft_total",
}

def normalize_households_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k:v for k,v in _RENAME.items() if k in df.columns}).copy()
    need = {"tract_geoid","group"}
    if not need.issubset(df.columns):
        raise ValueError(f"households file missing columns: {need - set(df.columns)}")
    df["tract_geoid"] = df["tract_geoid"].astype("string")
    df["group"] = df["group"].astype(str).str.lower().map({"owner":"owner","renter":"renter"}).fillna("renter")
    for c, default in (("rcv_kUSD",0.0),("contents_kUSD",0.0),("elev_ft_total",0.0)):
        if c not in df.columns: df[c]=default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default).astype(float)
    if "i" not in df.columns:
        df = df.reset_index(drop=True)
        df.insert(0,"i", df.index.values.astype(int)+1)
    if "i_orig" not in df.columns:
        df.insert(1,"i_orig", df["i"].astype(int))
    if "hh_idx_within_group" not in df.columns:
        df["hh_idx_within_group"] = df.groupby(["group","tract_geoid"]).cumcount()+1
    return df

def load_households_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".xlsx",".xls",".xlsm"):
        df = pd.read_excel(path, dtype={"tract_geoid":"string"})
    else:
        df = pd.read_csv(path, dtype={"tract_geoid":"string"})
    return normalize_households_columns(df)
# ---------- END Households helpers ----------