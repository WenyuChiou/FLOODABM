"""Run the paper's 50-draw Monte Carlo ensemble outside the repository.

The SFHA assignment, initial FI status, household attributes, flood sequence,
and depth-sampling RNG seed remain fixed. Runs vary the action-decision RNG seed
and Bayesian posterior draw only.
Running both scenarios produces 100 simulations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "config" / "abm_params.yaml"
N_RUNS = 50
N_POSTERIOR_DRAWS = 4800
FIRST_DECISION_SEED = 90001
DECISION_SEED_STEP = 37


def decision_seed(run_id: int) -> int:
    """Return the preregistered decision seed for a one-based run ID."""
    return FIRST_DECISION_SEED + DECISION_SEED_STEP * (run_id - 1)


def posterior_index(run_id: int) -> int:
    """Select evenly spaced posterior draws across the 4,800-sample chain."""
    return ((run_id - 1) * N_POSTERIOR_DRAWS) // N_RUNS


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [REPO / path, REPO / "modules" / path, REPO / "modules" / "actions" / path]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _sha256_files(paths: list[Path]) -> dict[str, object]:
    """Hash a sorted file set without embedding machine-specific paths."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        try:
            label = path.relative_to(REPO).as_posix()
        except ValueError:
            label = f"external/{path.name}"
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "n_files": len(paths)}


def reproducibility_fingerprint() -> dict[str, object]:
    """Fingerprint the code and scientific inputs that make reuse valid."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    files = config.get("files", {}) or {}
    sfha = config.get("sfha_initialization", {}) or {}
    required_inputs = {
        "config": CONFIG_PATH,
        "households": _resolve_repo_path(files["households"]),
        "mean_depths": _resolve_repo_path(files["depths_overall"]),
        "depth_distributions": _resolve_repo_path(files["depth_distributions"]),
        "sfha_shares": _resolve_repo_path(sfha["shares_file"]),
    }
    missing = [name for name, path in required_inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot fingerprint missing inputs: {missing}")
    flood_events = _resolve_repo_path(files.get("flood_events", ""))
    if files.get("flood_events") and flood_events.is_file():
        required_inputs["flood_events"] = flood_events

    code_paths = [REPO / "main.py", Path(__file__).resolve()]
    for directory in (REPO / "modules", REPO / "utils"):
        code_paths.extend(directory.rglob("*.py"))
    posterior_paths = sorted((REPO / "models" / "baseline").rglob("*.npz"))
    if not posterior_paths:
        raise FileNotFoundError("No Bayesian posterior .npz files in models/baseline")

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    return {
        "git_commit": git_commit,
        "code": _sha256_files(code_paths),
        "inputs": {
            name: _sha256_files([path]) for name, path in required_inputs.items()
        },
        "posterior_models": _sha256_files(posterior_paths),
    }


def build_design() -> dict[str, object]:
    """Return the complete formal-ensemble design and reuse fingerprint."""
    return {
        "n_runs": N_RUNS,
        "scenarios": ["baseline", "worst"],
        "fixed": [
            "SFHA assignment",
            "initial FI status",
            "household attributes",
            "historical flood sequence",
            "depth-sampling RNG seed",
        ],
        "varied": ["action-decision RNG seed", "Bayesian posterior draw"],
        "decision_seed_formula": "90001 + 37 * (run_id - 1)",
        "posterior_index_formula": "floor((run_id - 1) * 4800 / 50)",
        "fingerprints": reproducibility_fingerprint(),
    }


def validate_benchmark_config() -> None:
    """Fail before running if the active config is not the revised benchmark."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    checks = {
        "SFHA initialization": bool(
            (config.get("sfha_initialization", {}) or {}).get("enabled", False)
        ),
        "grid-depth sampling": (config.get("hazard", {}) or {}).get("depths_mode")
        == "grid_hist",
        "configured depth distributions": bool(
            (config.get("files", {}) or {}).get("depth_distributions")
        ),
        "renter FI interval": list((config.get("draw_bounds", {}) or {}).get("FI_renter", []))
        == [0.7, 0.9],
        "homeowner general FI interval": list(
            (config.get("draw_bounds", {}) or {}).get("FI", [])
        )
        == [0.35, 0.55],
        "inside-SFHA homeowner FI interval": list(
            (config.get("sfha_initialization", {}) or {}).get(
                "fi_draw_bounds_inside", []
            )
        )
        == [0.0, 0.1],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Active config is not the revised benchmark: {failed}")


def run_one(
    scenario: str,
    run_id: int,
    output_root: Path,
    *,
    timeout_seconds: float,
    reuse: bool,
) -> dict:
    """Run one immutable scenario/draw combination and return its audit row."""
    run_root = output_root / f"run_{run_id:02d}"
    result_dir = run_root / "baseline" / scenario
    summary_path = result_dir / "output_summary.json"
    log_dir = run_root / "logs"
    log_path = log_dir / f"{scenario}.log"
    if result_dir.exists():
        if reuse and summary_path.is_file():
            return {
                "scenario": scenario,
                "run_id": run_id,
                "decision_seed": decision_seed(run_id),
                "posterior_index": posterior_index(run_id),
                "status": "REUSED",
                "elapsed_s": 0.0,
                "result_dir": str(result_dir),
            }
        raise FileExistsError(
            f"Refusing to overwrite existing run output: {result_dir}"
        )
    log_dir.mkdir(parents=True, exist_ok=True)

    seed = decision_seed(run_id)
    draw = posterior_index(run_id)
    command = [
        sys.executable,
        "main.py",
        "--scenario",
        scenario,
        "--decision-seed",
        str(seed),
        "--output-mode",
        "summary",
        "--no-plots",
        "--out-root",
        str(run_root),
    ]
    environment = os.environ.copy()
    environment["FLOODABM_POSTERIOR_IDX"] = str(draw)
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            cwd=REPO,
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout_seconds,
            check=False,
        )
        status = "PASS" if process.returncode == 0 and summary_path.is_file() else "FAIL"
        log_path.write_text(
            "COMMAND\n"
            + subprocess.list2cmdline(command)
            + f"\n\nPOSTERIOR_INDEX\n{draw}\n\nSTDOUT\n{process.stdout}"
            + f"\n\nSTDERR\n{process.stderr}",
            encoding="utf-8",
        )
        return_code = process.returncode
    except subprocess.TimeoutExpired as exc:
        status = "TIMEOUT"
        return_code = None
        log_path.write_text(
            f"COMMAND\n{subprocess.list2cmdline(command)}\n\nTIMEOUT\n{exc}",
            encoding="utf-8",
        )
    elapsed = time.perf_counter() - started
    return {
        "scenario": scenario,
        "run_id": run_id,
        "decision_seed": seed,
        "posterior_index": draw,
        "status": status,
        "return_code": return_code,
        "elapsed_s": round(elapsed, 3),
        "result_dir": str(result_dir),
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["baseline", "worst"],
        default=["baseline", "worst"],
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=N_RUNS)
    parser.add_argument("--timeout-minutes", type=float, default=30.0)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.start <= args.end <= N_RUNS:
        parser.error(f"Require 1 <= start <= end <= {N_RUNS}")
    if args.timeout_minutes <= 0:
        parser.error("--timeout-minutes must be positive")

    output_root = args.out_root.resolve()
    if output_root == REPO or REPO in output_root.parents:
        parser.error("--out-root must be outside the repository")
    validate_benchmark_config()
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    status_path = output_root / "mc50_status.csv"
    design_path = output_root / "mc50_design.json"
    design = build_design()
    if design_path.exists():
        saved_design = json.loads(design_path.read_text(encoding="utf-8"))
        if saved_design != design:
            raise RuntimeError(f"Existing MC design does not match: {design_path}")
    else:
        if status_path.exists() or any(output_root.glob("run_*")):
            raise RuntimeError(
                "Existing Monte Carlo outputs have no matching design fingerprint; "
                "use a new --out-root."
            )
        design_path.write_text(json.dumps(design, indent=2), encoding="utf-8")

    existing_status = (
        pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    )
    for run_id in range(args.start, args.end + 1):
        for scenario in args.scenarios:
            row = run_one(
                scenario,
                run_id,
                output_root,
                timeout_seconds=args.timeout_minutes * 60.0,
                reuse=args.reuse,
            )
            rows.append(row)
            combined = pd.concat(
                [existing_status, pd.DataFrame(rows)], ignore_index=True
            )
            combined = (
                combined.drop_duplicates(["scenario", "run_id"], keep="last")
                .sort_values(["run_id", "scenario"])
                .reset_index(drop=True)
            )
            combined.to_csv(status_path, index=False, encoding="utf-8-sig")
            print(
                f"[{len(rows)}] {scenario} run_{run_id:02d}: {row['status']} "
                f"(seed={row['decision_seed']}, posterior={row['posterior_index']})",
                flush=True,
            )
            if row["status"] not in {"PASS", "REUSED"}:
                raise RuntimeError(f"Monte Carlo halted after {row['status']}: {row}")


if __name__ == "__main__":
    main()
