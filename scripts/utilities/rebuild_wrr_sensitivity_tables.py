"""Recompute WRR Tables S8–S9 from paired, fixed-initial-cohort finance totals."""

import argparse
import hashlib
from io import BytesIO
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .rebuild_wrr_finance_series import GROUPS, read_initial_counts, read_tract_finance
    from .verify_wrr_finance import ROOT, YEARS
else:
    from rebuild_wrr_finance_series import GROUPS, read_initial_counts, read_tract_finance
    from verify_wrr_finance import ROOT, YEARS


DATA = ROOT / "data" / "supplementary" / "wrr_s8_s9_cumulative_finance.csv"
DATA_SHA256 = "62b8f02bcc388bbd0c5e2f5bd66364e55d965c1af8d72faacbe1dd5a0642e13a"
CASES = ("uniform", "triangular", "truncated_normal", "same_year")
DRAWS = range(1, 11)
METRICS = ("GUL", "actual_loss")
KEYS = ["case", "draw", "group"]
VALUES = ["gross_usd", "payout_usd"]
STATS = ["median", "q25", "q75"]
ARCHIVES = {
    "triangular": (
        "08_threshold_distribution_sensitivity/raw/20260824_EX03_sfha_area_confirm10_01",
        "EX03_confirm_{draw:02d}/model_output/baseline/baseline/finance",
    ),
    "truncated_normal": (
        "08_threshold_distribution_sensitivity/raw/20260824_EX03_sfha_area_truncated_normal10_02",
        "EX03_confirm_{draw:02d}/model_output/baseline/baseline/finance",
    ),
    "same_year": (
        "04_decision_structure_sensitivity/raw/20260824_EX04_sfha_area_confirm10_01",
        "EX04_same_year_FI_first_EH/run_{draw:02d}/model_output/baseline/baseline/finance",
    ),
}
RAW_SHA256 = {
    "uniform": "1ab6b9b5d65118277c26e0ae8078165acbaf1332d9352169f15c164f4dd1a8cf",
    "triangular": "b4bf5d703897ee313d3002de544521b991f0870224151c934f5b089b643a7753",
    "truncated_normal": "d0768f632e480a461f23f1bd215fe12ea0c91178de42ff6383ad0244fc1c1813",
    "same_year": "004d34c93547da631a3d74ff2f0ce14dc83fbfcfa7d6ad42dc6818a1eecd7bd2",
}
# The final SM and response print the median followed by the two IQR endpoints.
DISPLAY = {
    "triangular": (
        ("6.3", "6.2", "8.1"), ("6.0", "5.7", "7.2"),
        ("3.4", "3.2", "3.6"), ("2.0", "1.9", "2.2"),
    ),
    "truncated_normal": (
        ("9.8", "8.3", "11.0"), ("8.6", "7.7", "10.7"),
        ("5.4", "5.1", "6.0"), ("3.1", "2.9", "3.4"),
    ),
    "same_year": (
        ("-59.9", "-63.1", "-57.0"), ("-6.9", "-7.7", "-5.2"),
        ("-0.5", "-0.8", "-0.2"), ("0.7", "0.5", "0.8"),
    ),
}


def read_raw_totals(archive: Path, mc: Path, initial: dict[str, int]) -> pd.DataFrame:
    rows = []
    for case in CASES:
        if case == "uniform":
            root, pattern = mc, "run_{draw:02d}/baseline/baseline/finance"
        else:
            directory, pattern = ARCHIVES[case]
            root = archive / directory
        digest = hashlib.sha256()
        for draw in DRAWS:
            totals = {group: [] for group in GROUPS}
            for year in YEARS:
                path = root / pattern.format(draw=draw) / f"finance_tract_{year}.csv"
                payload = path.read_bytes().replace(b"\r\n", b"\n")
                digest.update(f"{draw:02d}/{year}\0".encode("ascii") + payload + b"\0")
                tenure, _ = read_tract_finance(path, payload=payload)
                for group, data in tenure.items():
                    if year == YEARS.start and data["households"] != initial[group]:
                        raise ValueError(f"Wrong initial {group} population for {case} draw {draw}")
                    totals[group].append((data["gul"], data["payout"]))
            for group, values in totals.items():
                gross, payout = (fsum(column) for column in zip(*values))
                rows.append((case, draw, group, gross, payout))
        if digest.hexdigest() != RAW_SHA256[case]:
            raise ValueError(f"Raw annual finance files do not match the revised {case} case")
    return pd.DataFrame(rows, columns=KEYS + VALUES)


def validate_totals(totals: pd.DataFrame) -> pd.DataFrame:
    indexed = totals.set_index(KEYS, verify_integrity=True).sort_index()
    expected = pd.MultiIndex.from_product([CASES, DRAWS, GROUPS], names=KEYS).sort_values()
    if not indexed.index.equals(expected):
        raise ValueError("Finance totals must include every revised case, matched draw, and tenure")
    values = indexed[VALUES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values[:, 1] > values[:, 0]).any():
        raise ValueError("Finance totals must be nonnegative and payouts must not exceed GUL")
    return indexed


def summarize(totals: pd.DataFrame, initial: dict[str, int]) -> pd.DataFrame:
    indexed = validate_totals(totals)
    base = indexed.xs("uniform")[VALUES]
    rows = []
    for case in CASES[1:]:
        delta = indexed.xs(case)[VALUES] - base
        for group in GROUPS:
            group_delta = delta.xs(group, level="group")
            gross = group_delta["gross_usd"]
            actual = gross - group_delta["payout_usd"]
            for metric, difference in zip(METRICS, (gross, actual)):
                normalized = difference / (1000 * initial[group])
                rows.append(("S9" if case == "same_year" else "S8", case, group, metric,
                             normalized.median(), normalized.quantile(.25), normalized.quantile(.75)))
    return pd.DataFrame(rows, columns=["table", "case", "group", "metric", *STATS])


def verify_display(summary: pd.DataFrame) -> None:
    for case, expected in DISPLAY.items():
        values = summary.loc[summary["case"].eq(case), STATS]
        actual = tuple(tuple(f"{value:.1f}" for value in row)
                       for row in values.itertuples(index=False, name=None))
        if actual != expected:
            raise ValueError(f"{case} results do not match the submitted one-decimal cells")


def read_bundled() -> pd.DataFrame:
    # Normalize only line endings so a Windows Git checkout keeps the same pin.
    payload = DATA.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
        raise ValueError("Bundled combined finance totals do not match the revised reference")
    return pd.read_csv(BytesIO(payload), float_precision="round_trip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, help="Root of the separately archived sensitivity cases")
    parser.add_argument("--mc-root", type=Path, help="Main ensemble; the first ten adaptation draws are controls")
    parser.add_argument("--out-dir", type=Path, help="New directory for the recomputed S8/S9 CSVs")
    args = parser.parse_args()
    if (args.archive_root is None) != (args.mc_root is None):
        parser.error("--archive-root and --mc-root must be provided together")
    if args.out_dir is not None and (args.out_dir.exists() or args.out_dir.is_symlink()):
        parser.error("--out-dir must not exist")
    try:
        initial = read_initial_counts()
        bundled = read_bundled()
        if args.archive_root is None:
            totals = bundled
        else:
            totals = read_raw_totals(args.archive_root, args.mc_root, initial)
            # The bundled totals are rounded to cents, far below the table precision.
            delta = validate_totals(totals)[VALUES] - validate_totals(bundled)[VALUES]
            if delta.abs().to_numpy().max() > .00501:
                raise ValueError("Raw reconstruction differs from the bundled cumulative totals")
            print("PASS: 520 annual tract finance files match the audited four-case archive")
        summary = summarize(totals, initial)
        verify_display(summary)
        print("PASS: Tables S8–S9 match all 12 submitted cells (36 displayed values)")
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True)
            for table, rows in summary.groupby("table"):
                rows.to_csv(args.out_dir / f"Table_{table}.csv", index=False, mode="x",
                            lineterminator="\n", float_format="%.1f")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
