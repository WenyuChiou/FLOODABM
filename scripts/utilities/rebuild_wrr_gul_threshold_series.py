"""Verify and rebuild the tract GUL ratio threshold sensitivity for WRR Figure S5."""

import argparse
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .rebuild_wrr_finance_series import FINANCE_REFERENCE, read_initial_counts, read_tract_finance
    from .rebuild_wrr_sequence_series import publish_tables
    from .verify_wrr_finance import YEARS, verify
    from .wrr_threshold_cases import (
        ControlPayloads, DRAW_IDS, S5_TENURE as TENURE, S5_THRESHOLDS, bundled_plot,
        case_output_root, load_hashes, snapshot_controls, verified_finance,
    )
else:
    from rebuild_wrr_finance_series import FINANCE_REFERENCE, read_initial_counts, read_tract_finance
    from rebuild_wrr_sequence_series import publish_tables
    from verify_wrr_finance import YEARS, verify
    from wrr_threshold_cases import (
        ControlPayloads, DRAW_IDS, S5_TENURE as TENURE, S5_THRESHOLDS, bundled_plot,
        case_output_root, load_hashes, snapshot_controls, verified_finance,
    )


ANNUAL = "annual_metrics.csv"
PAIRED = "paired_fixed_initial_by_run.csv"
PLOT = "Figure_R2_fixed_initial_data.csv"
PINNED = {
    ANNUAL: "b85a8b9258cdd880689869818d46f2d13f8b8ee4837e7650d68575dd09244665",
    PAIRED: "33c0306cefaa085c1ff20c0f7a6de9b22319d1e8abe9882bf1034aa1ab378c4c",
    PLOT: "b4d926f05068b576c3fbb53839c9f001f6625cdad24b2bf858a955e978f912d7",
}
CONTROL_ID = "EX02_control"
LOSSES = ["GUL", "actual_loss"]
SUMMED = [f"{name}_total_usd" for name in LOSSES]
PAIR_KEYS = ["case_id", "run_number", "group", "series", "threshold"]
ANNUAL_KEYS = [*PAIR_KEYS, "year"]
PLOT_KEYS = ["case_id", "series", "group", "threshold"]
PLOT_STATS = [f"{name}_{stat}" for name in LOSSES for stat in ("median", "q1", "q3")]
PAIR_STATS = [*SUMMED, *LOSSES, *(f"{n}_control" for n in LOSSES),
              *(f"{n}_difference" for n in LOSSES)]


def archived_table(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PINNED[name]:
        raise ValueError(f"Checksum differs from the audited Figure S5 source: {path}")
    return pd.read_csv(BytesIO(payload), float_precision="round_trip")


def rebuild_annual(case_root: Path, controls: Path,
                   control_payloads: ControlPayloads,
                   checksums: dict[tuple[str, int], str] | None = None) -> pd.DataFrame:
    cases = [("control", .10, CONTROL_ID)]
    cases.extend((series, code / 100, f"EX02_{series}_{code:03d}")
                 for series in TENURE for code in S5_THRESHOLDS)
    rows = []
    for draw in DRAW_IDS:
        for series, threshold, case_id in cases:
            payloads = None
            if case_id == CONTROL_ID:
                finance = controls / f"run_{draw:02d}" / "baseline" / "baseline" / "finance"
                payloads = control_payloads[draw]
            else:
                finance = case_root / case_output_root(5, case_id, draw) / "baseline" / "baseline" / "finance"
                if checksums is not None:
                    payloads = verified_finance(case_root, 5, case_id, draw, checksums[case_id, draw])
            for year in YEARS:
                tenure, _ = read_tract_finance(finance / f"finance_tract_{year}.csv",
                                               payload=None if payloads is None else payloads[year])
                for group, totals in tenure.items():
                    rows.append((case_id, draw, group, series, threshold, year,
                                 totals["households"], totals["gul"], totals["gul"] - totals["payout"]))
    return pd.DataFrame(rows, columns=[*ANNUAL_KEYS, "households", *SUMMED])


def summarize(annual: pd.DataFrame, initial: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    starts = annual.loc[annual["year"].eq(YEARS.start), ["group", "households"]]
    if not starts["households"].eq(starts["group"].map(initial)).all():
        raise ValueError("A threshold case has a different initial tenure population")
    total = annual.groupby(PAIR_KEYS, as_index=False)[SUMMED].sum()
    for loss in LOSSES:
        total[loss] = total[f"{loss}_total_usd"] / total["group"].map(initial)
    control = total.loc[total["case_id"].eq(CONTROL_ID), ["run_number", "group", *LOSSES]]
    cases = total.loc[total["series"].map(TENURE).eq(total["group"])]
    paired = cases.merge(control, on=["run_number", "group"], how="left",
                         suffixes=("", "_control"), validate="many_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("A threshold case lacks a same-draw, same-tenure control")
    paired = paired.drop(columns="_merge")
    for loss in LOSSES:
        paired[f"{loss}_difference"] = paired[loss] - paired[f"{loss}_control"]
    observed = paired.groupby(PLOT_KEYS)
    stats = {}
    for loss in LOSSES:
        diff = observed[f"{loss}_difference"]
        stats.update({f"{loss}_median": diff.median(), f"{loss}_q1": diff.quantile(.25),
                      f"{loss}_q3": diff.quantile(.75)})
    plot = pd.DataFrame(stats).reset_index().sort_values(["series", "threshold"])
    return paired, plot


def compare(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str],
            values: list[str], tolerance: float) -> float:
    a = actual.set_index(keys, verify_integrity=True).sort_index()
    b = expected.set_index(keys, verify_integrity=True).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Sensitivity row keys differ from the accepted archive")
    delta = (a[values] - b[values]).abs().to_numpy()
    if not np.isfinite(delta).all() or delta.max() > tolerance:
        raise ValueError(f"Sensitivity values differ from the accepted archive (limit {tolerance:g})")
    return float(delta.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True, help="Replayed or archived 160 sensitivity cases")
    parser.add_argument("--mc-root", type=Path, required=True, help="Verified main ensemble; first ten are controls")
    parser.add_argument("--reference-dir", type=Path, help="Use the private pinned r2 references (archive mode)")
    parser.add_argument("--out-dir", type=Path, help="New directory for verified plot and per-draw CSVs")
    args = parser.parse_args()
    if args.out_dir is not None and (args.out_dir.exists() or args.out_dir.is_symlink()):
        parser.error("--out-dir must not exist")
    try:
        verify(args.mc_root, FINANCE_REFERENCE, 1, 10)
        controls = snapshot_controls(args.mc_root, FINANCE_REFERENCE)
        if args.reference_dir is not None:
            annual, paired_expected, plot_expected = (
                archived_table(args.reference_dir, n) for n in (ANNUAL, PAIRED, PLOT)
            )
            actual = rebuild_annual(args.case_root, args.mc_root, controls)
            delta = compare(actual, annual, ANNUAL_KEYS, ["households", *SUMMED], 1e-5)
            if not actual.set_index(ANNUAL_KEYS)["households"].sort_index().equals(
                annual.set_index(ANNUAL_KEYS)["households"].astype(float).sort_index()
            ):
                raise ValueError("A Figure S5 annual tenure count differs from the accepted archive")
            print(f"PASS: {len(actual):,} annual tenure records; max difference {delta:.3g} USD")
        else:
            plot_expected = pd.read_csv(BytesIO(bundled_plot(5)), float_precision="round_trip")
            actual = rebuild_annual(args.case_root.resolve(strict=True), args.mc_root, controls, load_hashes(5))
            print("Verified 2,080 variation finance files plus the paired main-ensemble controls.")
        paired, plot = summarize(actual, read_initial_counts())
        if args.reference_dir is not None:
            raw_delta = compare(paired, paired_expected, PAIR_KEYS, SUMMED, 1e-5)
            paired_delta = compare(paired, paired_expected, PAIR_KEYS, PAIR_STATS[len(SUMMED):], 1e-8)
            print(f"PASS {PAIRED}: {len(paired)} rows, max total {raw_delta:.3g} USD, "
                  f"max per household {paired_delta:.3g} USD")
        plot_delta = compare(plot, plot_expected, PLOT_KEYS, PLOT_STATS, 1e-8)
        print(f"PASS {PLOT}: {len(plot)} rows, max {plot_delta:.3g} USD/household")
        if args.out_dir is not None:
            publish_tables(args.out_dir, paired[[*PAIR_KEYS, *PAIR_STATS]], plot[[*PLOT_KEYS, *PLOT_STATS]],
                           names=(PAIRED, PLOT))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
