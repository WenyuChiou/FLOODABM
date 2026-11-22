"""Replay and verify the revised WRR Figure S6 simulations from a clone."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .rebuild_wrr_sequence_series import (
        CASES, DRAWS, ORDER_IDS, ROOT, SCENARIOS, confined_path, read_public_hashes,
        read_verified_case,
    )
    from .run_mc100_local import decision_seed, posterior_index, validate_benchmark_config
else:
    from rebuild_wrr_sequence_series import (
        CASES, DRAWS, ORDER_IDS, ROOT, SCENARIOS, confined_path, read_public_hashes,
        read_verified_case,
    )
    from run_mc100_local import decision_seed, posterior_index, validate_benchmark_config


def run_case(suite: Path, case: tuple[str, int, str], expected: str, timeout: float, reuse: bool) -> str:
    order, draw, scenario = case
    target = confined_path(suite, Path("raw") / order / f"draw_{draw:02d}" / scenario)
    marker = target / "verified.sha256"
    if target.exists():
        if not reuse:
            raise FileExistsError(f"Case directory already exists: {target}")
        marker = confined_path(suite, marker.relative_to(suite))
        if marker.read_text(encoding="utf-8").strip() != expected:
            raise ValueError(f"Case lacks the expected successful replay marker: {target}")
        read_verified_case(suite, case, expected)
        return "VERIFIED EXISTING"

    target.mkdir(parents=True)
    command = [
        sys.executable, "main.py", "--scenario", scenario, "--model-dir", "baseline",
        "--decision-seed", str(decision_seed(draw)), "--output-mode", "summary",
        "--no-plots", "--wrr-flood-order", order, "--out-root", str(target),
    ]
    environment = dict(os.environ, FLOODABM_POSTERIOR_IDX=str(posterior_index(draw)))
    log_path = target / "run.log"
    with log_path.open("xb") as log:
        log.write((json.dumps(command) + "\n"
                   f"FLOODABM_POSTERIOR_IDX={posterior_index(draw)}\n").encode("utf-8"))
        log.flush()
        try:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, timeout=timeout, check=True)
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(f"Case {case} did not finish; see {log_path}") from exc
    read_verified_case(suite, case, expected)
    with marker.open("x", encoding="utf-8") as handle:
        handle.write(expected + "\n")
    return "REPLAYED AND VERIFIED"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True, help="Output directory outside the clone")
    parser.add_argument("--orders", nargs="+", choices=ORDER_IDS, default=ORDER_IDS)
    parser.add_argument("--draws", nargs="+", type=int, choices=DRAWS, default=DRAWS)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--timeout-seconds", type=float, default=1800, help="Timeout per case")
    parser.add_argument("--reuse", action="store_true", help="Verify and skip existing cases")
    args = parser.parse_args()
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    selected = [case for case in CASES if case[0] in args.orders and case[1] in args.draws
                and case[2] in args.scenarios]
    try:
        suite = args.suite_root.resolve()
        if suite.is_relative_to(ROOT):
            raise ValueError("--suite-root must be outside the repository")
        validate_benchmark_config()
        hashes = read_public_hashes()
        for index, case in enumerate(selected, start=1):
            order, draw, scenario = case
            print(f"[{index}/{len(selected)}] {order} draw {draw} {scenario}", flush=True)
            status = run_case(suite, case, hashes[case], args.timeout_seconds, args.reuse)
            print(f"  {status}: {case}", flush=True)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
