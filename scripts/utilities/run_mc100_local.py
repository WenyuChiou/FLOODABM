# -*- coding: utf-8 -*-
"""
Monte Carlo 100-run (baseline + worst) using the legacy mc_initpsy seeds.

- Reads seeds from data/supplementary/mc_initpsy_seeds.csv (50 rows).
- Patches insurance_init.seed (init_seed) and top-level seed (decision_seed).
- Runs main.py twice per seed: --scenario baseline then --scenario worst.
- Writes each run under FLOODABM_MC_ROOT (default outputs/mc50/) as
  {scenario}/run_XX/baseline/{scenario}/...
- Deletes the large `states/` subfolder after each run to keep disk usage down.
- Always restores the original YAML on exit (success or failure).

Usage:
    python scripts/utilities/run_mc100_local.py
    python scripts/utilities/run_mc100_local.py --scenarios baseline
    python scripts/utilities/run_mc100_local.py --start 1 --end 5   # test slice

Bayesian posterior sampling:
    Each run sets FLOODABM_POSTERIOR_IDX so the decision model draws a distinct
    posterior sample per run (see modules/actions/bayes_fast_predictors.py,
    Priority 0). The posterior-sample .npz (posterior_beta_* arrays with their
    calibrators) ship in models/baseline/, so the 50-run posterior spread is
    reproduced directly. With FLOODABM_POSTERIOR_IDX unset, runs use the
    posterior-mean cache in models/baseline_fast/.
"""
from __future__ import annotations
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
YAML_PATH = REPO / "config" / "abm_params.yaml"
# Seed list is published under data/supplementary/ (tracked). Fall back to the
# legacy outputs/montecarlo/ location if a local copy is present there.
SEEDS_CSV = REPO / "data" / "supplementary" / "mc_initpsy_seeds.csv"
if not SEEDS_CSV.exists():
    SEEDS_CSV = REPO / "outputs" / "montecarlo" / "mc_initpsy_seeds.csv"
# Output root is configurable via the FLOODABM_MC_ROOT environment variable and
# defaults to a repo-relative folder, so the script is portable (this was a
# hardcoded C:\ path).
OUT_ROOT = Path(os.environ.get("FLOODABM_MC_ROOT", str(REPO / "outputs" / "mc50")))


def patch_yaml(text: str, init_seed: int, decision_seed: int) -> str:
    text = re.sub(
        r"(insurance_init:\s*\n\s*seed:\s*)\d+",
        lambda m: f"{m.group(1)}{init_seed}",
        text,
        count=1,
    )
    text = re.sub(
        r"^(seed:\s*)\d+",
        lambda m: f"{m.group(1)}{decision_seed}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    return text


def run_one(scenario: str, run_id: int, init_seed: int, decision_seed: int,
            original_yaml: str) -> tuple[bool, float]:
    run_dir = OUT_ROOT / scenario / f"run_{run_id:02d}"
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    patched = patch_yaml(original_yaml, init_seed, decision_seed)
    YAML_PATH.write_text(patched, encoding="utf-8")

    # Bayesian posterior sampling: each run uses a distinct posterior sample
    # index so that the 50 MC runs also propagate the regression coefficient
    # uncertainty (4800 total samples available; we sample every 96-th one).
    posterior_idx = ((run_id - 1) * 4800) // 50  # uniform stride across chain
    env = os.environ.copy()
    env["FLOODABM_POSTERIOR_IDX"] = str(posterior_idx)

    t0 = time.time()
    proc = subprocess.run(
        [
            sys.executable, "main.py",
            "--scenario", scenario,
            "--output-mode", "summary",
            "--no-plots",
            "--out-root", str(run_dir),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        log = run_dir / "run_failed.log"
        log.write_text(
            f"returncode={proc.returncode}\n\nSTDOUT:\n{proc.stdout[-4000:]}\n\nSTDERR:\n{proc.stderr[-4000:]}",
            encoding="utf-8",
        )
        return False, elapsed

    # Drop the huge states/ folder — not needed for any figure.
    states_dir = run_dir / "baseline" / scenario / "states"
    if states_dir.exists():
        shutil.rmtree(states_dir, ignore_errors=True)

    return True, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["baseline", "worst"],
                    choices=["baseline", "worst"])
    ap.add_argument("--start", type=int, default=1, help="first run_id (inclusive)")
    ap.add_argument("--end", type=int, default=50, help="last run_id (inclusive)")
    args = ap.parse_args()

    seeds_df = pd.read_csv(SEEDS_CSV)
    seeds_df.columns = [c.lstrip("\ufeff") for c in seeds_df.columns]
    seeds_df = seeds_df[(seeds_df["run_id"] >= args.start) & (seeds_df["run_id"] <= args.end)].reset_index(drop=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    original_yaml = YAML_PATH.read_text(encoding="utf-8")

    progress_path = OUT_ROOT / "mc100_progress.csv"
    results = []
    t_total = time.time()
    n_total = len(seeds_df) * len(args.scenarios)
    done = 0

    try:
        for scenario in args.scenarios:
            print(f"\n{'='*60}\n[scenario] {scenario}\n{'='*60}")
            for _, row in seeds_df.iterrows():
                run_id = int(row["run_id"])
                init_seed = int(row["init_seed"])
                decision_seed = int(row["decision_seed"])

                done += 1
                prefix = f"[{done}/{n_total}] {scenario} run_{run_id:02d} (init={init_seed}, dec={decision_seed})"
                print(prefix, flush=True)
                ok, elapsed = run_one(scenario, run_id, init_seed, decision_seed, original_yaml)
                status = "OK" if ok else "FAIL"
                print(f"    -> {status} in {elapsed:.1f}s", flush=True)

                results.append({
                    "scenario": scenario, "run_id": run_id,
                    "init_seed": init_seed, "decision_seed": decision_seed,
                    "ok": ok, "elapsed_s": round(elapsed, 2),
                })
                pd.DataFrame(results).to_csv(progress_path, index=False, encoding="utf-8-sig")
    finally:
        YAML_PATH.write_text(original_yaml, encoding="utf-8")
        print(f"\n[yaml] restored original to {YAML_PATH}")

    total_min = (time.time() - t_total) / 60.0
    df = pd.DataFrame(results)
    n_ok = int(df["ok"].sum()) if len(df) else 0
    print(f"\n{'='*60}\nDone: {n_ok}/{len(df)} runs OK in {total_min:.1f} min")
    print(f"Progress log: {progress_path}")


if __name__ == "__main__":
    main()
