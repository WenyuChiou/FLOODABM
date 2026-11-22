"""Replay revised Figure S4 or S5 tests, with per-draw finance verification."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .run_mc100_local import decision_seed, posterior_index, validate_benchmark_config
    from .wrr_threshold_cases import (
        CASE_IDS, DRAW_IDS, ROOT, case_model_args, case_output_root, confined_path,
        load_hashes, verified_finance,
    )
else:
    from run_mc100_local import decision_seed, posterior_index, validate_benchmark_config
    from wrr_threshold_cases import (
        CASE_IDS, DRAW_IDS, ROOT, case_model_args, case_output_root, confined_path,
        load_hashes, verified_finance,
    )


def replay_case(root: Path, number: int, case: str, draw: int,
                expected: str, timeout: float, reuse: bool) -> str:
    relative = case_output_root(number, case, draw)
    target = confined_path(root, relative)
    marker = target / "verified.sha256"
    if target.exists():
        if not reuse:
            raise FileExistsError(f"Case directory already exists: {target}")
        marker = confined_path(root, relative / marker.name)
        if marker.read_text(encoding="ascii").strip() != expected:
            raise ValueError(f"Case lacks the expected successful replay marker: {target}")
        verified_finance(root, number, case, draw, expected)
        return "VERIFIED EXISTING"

    target.mkdir(parents=True)
    command = [
        sys.executable, "main.py", "--scenario", "baseline", "--model-dir", "baseline",
        "--decision-seed", str(decision_seed(draw)), "--output-mode", "summary", "--no-plots",
        "--out-root", str(target), *case_model_args(number, case),
    ]
    posterior = str(posterior_index(draw))
    environment = dict(os.environ, FLOODABM_POSTERIOR_IDX=posterior)
    log_path = target / "run.log"
    with log_path.open("xb") as log:
        log.write(f"{json.dumps(command)}\nFLOODABM_POSTERIOR_IDX={posterior}\n".encode("utf-8"))
        log.flush()
        try:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, timeout=timeout, check=True)
        except (subprocess.SubprocessError, OSError) as error:
            raise RuntimeError(f"{case}, draw {draw} did not finish; see {log_path}") from error
    verified_finance(root, number, case, draw, expected)
    with marker.open("x", encoding="ascii") as handle:
        handle.write(expected + "\n")
    return "REPLAYED AND VERIFIED"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=tuple(CASE_IDS), required=True)
    parser.add_argument("--case-root", type=Path, required=True, help="Output directory outside the clone")
    parser.add_argument("--cases", nargs="+", help="Run only these saved case identifiers")
    parser.add_argument("--draws", type=int, nargs="+", choices=DRAW_IDS, default=DRAW_IDS)
    parser.add_argument("--timeout-seconds", type=float, default=1800, help="Timeout per case")
    parser.add_argument("--reuse", action="store_true", help="Verify and skip successfully completed cases")
    args = parser.parse_args()
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    if args.cases and set(args.cases).difference(CASE_IDS[args.figure]):
        parser.error("--cases must name saved case identifiers for the selected figure")
    all_cases = CASE_IDS[args.figure]
    chosen = [(case, draw) for case in all_cases for draw in DRAW_IDS
              if (args.cases is None or case in args.cases) and draw in args.draws]
    try:
        root = args.case_root.resolve()
        if root.is_relative_to(ROOT):
            raise ValueError("--case-root must be outside the repository")
        validate_benchmark_config()
        hashes = load_hashes(args.figure)
        for index, (case, draw) in enumerate(chosen, start=1):
            print(f"[{index}/{len(chosen)}] {case}, draw {draw}", flush=True)
            result = replay_case(root, args.figure, case, draw, hashes[case, draw],
                                 args.timeout_seconds, args.reuse)
            print(f"  {result}: {case}, draw {draw}", flush=True)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
