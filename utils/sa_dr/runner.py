# utils/sa_dr/runner.py
from __future__ import annotations
from pathlib import Path
import os, sys, time, subprocess, shutil, copy, json, yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DEFAULT = ROOT / "config" / "abm_params.yaml"
OUT_ROOT = ROOT / "outputs"
EXP_ROOT = OUT_ROOT / "experiments"

def _load_cfg(p: Path) -> dict:
    txt = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(txt) or {}
    return json.loads(txt)

def _deep_update(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def run_experiment(exp_name: str, cfg_overrides: dict, output_mode: str = "full") -> Path:
    dst = EXP_ROOT / exp_name
    dst.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["FLOODABM_OUTPUT_ROOT"] = str(dst)
    env["FLOODABM_OUTPUT_MODE"] = output_mode  # Pass output mode to subprocess

    # Extract flood threshold overrides
    thr_mg  = cfg_overrides.get("flood", {}).get("ratio_threshold_MG", None)
    thr_nmg = cfg_overrides.get("flood", {}).get("ratio_threshold_NMG", None)
    
    # Extract tp_shock overrides
    shock_owner = cfg_overrides.get("tp_shock", {}).get("shock_scale_owner", None)
    shock_renter = cfg_overrides.get("tp_shock", {}).get("shock_scale_renter", None)

    cmd = [sys.executable, "main.py", "--scenario", "baseline"]
    if thr_mg is not None:
        cmd += ["--thr-mg", str(thr_mg)]
    if thr_nmg is not None:
        cmd += ["--thr-nmg", str(thr_nmg)]
    if shock_owner is not None:
        cmd += ["--shock-owner", str(shock_owner)]
    if shock_renter is not None:
        cmd += ["--shock-renter", str(shock_renter)]

    subprocess.check_call(cmd, cwd=str(ROOT), env=env)

    if not any(dst.iterdir()):
        raise RuntimeError(f"main.py finished but no files in {dst}")
    return dst

