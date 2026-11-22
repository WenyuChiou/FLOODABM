"""Verify the tract finance outputs against the archived revised ensemble."""

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YEARS = range(2011, 2024)
SCENARIOS = ("baseline", "worst")
N_RUNS = 50


def finance_digest(output: Path, run_id: int, scenario: str) -> str:
    finance = output / f"run_{run_id:02d}" / "baseline" / scenario / "finance"
    digest = hashlib.sha256()
    for year in YEARS:
        name = f"finance_tract_{year}.csv"
        payload = (finance / name).read_bytes().replace(b"\r\n", b"\n")
        digest.update(name.encode("ascii") + b"\0" + payload + b"\0")
    return digest.hexdigest()


def verify(output: Path, reference: Path, start: int, end: int) -> int:
    with reference.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(int(r["run_id"]), r["scenario"]): r["finance_sha256"] for r in rows}
    keys = {(i, scenario) for i in range(1, N_RUNS + 1) for scenario in SCENARIOS}
    if len(rows) != len(keys) or expected.keys() != keys:
        raise ValueError("Reference must contain each run and scenario exactly once")
    checked = 0
    for run_id in range(start, end + 1):
        for scenario in SCENARIOS:
            actual = finance_digest(output, run_id, scenario)
            if actual != expected[(run_id, scenario)]:
                raise ValueError(f"Archived finance output differs: run {run_id}, {scenario}")
            checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=N_RUNS)
    args = parser.parse_args()
    if not 1 <= args.start <= args.end <= N_RUNS:
        parser.error("require 1 <= start <= end <= 50")
    reference = ROOT / "data" / "supplementary" / "wrr_mc50_finance_reference.csv"
    n = verify(args.mc_root, reference, args.start, args.end)
    print(f"Verified {n} scenario runs ({n * len(YEARS)} annual tract finance CSVs).")


if __name__ == "__main__":
    main()
