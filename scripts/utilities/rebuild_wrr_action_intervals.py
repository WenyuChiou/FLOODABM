"""Verify and rebuild the action threshold interval sensitivity for WRR Figure S4."""

import argparse
import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd

if __package__:
    from .rebuild_wrr_finance_series import FINANCE_REFERENCE, read_initial_counts, read_tract_finance
    from .rebuild_wrr_gul_threshold_series import DRAW_IDS, LOSSES, PLOT_STATS, SUMMED, compare
    from .verify_wrr_finance import YEARS, verify
else:
    from rebuild_wrr_finance_series import FINANCE_REFERENCE, read_initial_counts, read_tract_finance
    from rebuild_wrr_gul_threshold_series import DRAW_IDS, LOSSES, PLOT_STATS, SUMMED, compare
    from verify_wrr_finance import YEARS, verify


CONTROL = "EX01_control"
COMPONENT = {"FI_homeowner_outside_SFHA": "owner", "FI_renter": "renter",
             "EH": "owner", "BP": "owner", "RL": "renter"}
VARIATIONS = ("lower", "higher", "narrower", "wider")
REGISTRY_HASH = "e0b2339241f03036d2eceb2fa1be2b448af084dae1216d391858a138d7d97499"
ANNUAL_HASH = "9f21eada959eb409cb038ce91bcc6bdcadacdaee07c3e4251dddf530bb95efca"
PAIRED = "paired_outcomes_by_run.csv"
PAIRED_HASH = "60b9654b42733aeaf4c1ea4ccc292c5e21f1811cc5935d7188581eeb9ccf7c7d"
PLOT = "Figure_R1_data.csv"
PLOT_HASH = "088bfee6c30ab1eb780bc02fc16e4e1961df85d3351340348f15cfd20ea31484"
PAIR_KEYS = ["case_id", "run_number", "group"]
ANNUAL_KEYS = [*PAIR_KEYS, "year"]
PLOT_KEYS = ["case_id", "group", "case_order", "component", "variation", "lower_bound", "upper_bound"]
PAIR_STATS = ["initial_households", *LOSSES, *(f"{n}_control" for n in LOSSES),
              *(f"{n}_difference" for n in LOSSES)]


def audited_csv(path: Path, digest: str) -> pd.DataFrame:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"Checksum differs from the accepted Figure S4 archive: {path}")
    return pd.read_csv(BytesIO(payload), float_precision="round_trip")


def case_finance(path: Path) -> Path:
    candidates = [p for p in (path / "finance", path / "baseline" / "finance") if p.is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one recognized archive finance folder under {path}")
    return candidates[0]


def rebuild_annual(case_root: Path, control_root: Path, registry: pd.DataFrame) -> pd.DataFrame:
    case_ids = {f"EX01_{component}_{variant}" for component in COMPONENT for variant in VARIATIONS}
    if len(registry) != len(case_ids) or set(registry["case_id"]) != case_ids:
        raise ValueError("Expected all 20 audited interval cases")
    rows = []
    for case in (CONTROL, *sorted(case_ids)):
        for draw in DRAW_IDS:
            if case == CONTROL:
                finance = control_root / f"run_{draw:02d}" / "baseline" / "baseline" / "finance"
            else:
                finance = case_finance(case_root / case / f"run_{draw:02d}" / "model_output" / "baseline" / "baseline")
            for year in YEARS:
                tenure, _ = read_tract_finance(finance / f"finance_tract_{year}.csv")
                for group, totals in tenure.items():
                    rows.append((case, draw, group, year, totals["households"], totals["gul"],
                                 totals["gul"] - totals["payout"]))
    return pd.DataFrame(rows, columns=[*ANNUAL_KEYS, "households", *SUMMED])


def summarize(annual: pd.DataFrame, registry: pd.DataFrame,
              initial: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = annual.loc[annual["year"].eq(YEARS.start), [*PAIR_KEYS, "households"]]
    if not baseline["households"].eq(baseline["group"].map(initial)).all():
        raise ValueError("An action-interval case has a different initial tenure population")
    totals = annual.groupby(PAIR_KEYS, as_index=False)[SUMMED].sum()
    totals = totals.merge(baseline.rename(columns={"households": "initial_households"}),
                          on=PAIR_KEYS, validate="one_to_one", how="left")
    if totals["initial_households"].isna().any():
        raise ValueError("An action-interval case lacks a first-year population")
    for loss in LOSSES:
        totals[loss] = totals[f"{loss}_total_usd"] / totals["initial_households"]
    control = totals.loc[totals["case_id"].eq(CONTROL), [*PAIR_KEYS[1:], *LOSSES]]
    paired = totals.merge(control, on=PAIR_KEYS[1:], suffixes=("", "_control"),
                          how="left", validate="many_to_one")
    if paired[[f"{n}_control" for n in LOSSES]].isna().any().any():
        raise ValueError("An action-interval case lacks a same-draw, same-tenure control")
    for loss in LOSSES:
        paired[f"{loss}_difference"] = paired[loss] - paired[f"{loss}_control"]

    metadata = registry[[k for k in PLOT_KEYS if k != "group"]].copy()
    metadata["group"] = metadata["component"].map(COMPONENT)
    plotted = paired.merge(metadata, on=["case_id", "group"], validate="many_to_one")
    observed = plotted.groupby(PLOT_KEYS)
    stats = {}
    for loss in LOSSES:
        diff = observed[f"{loss}_difference"]
        stats.update({f"{loss}_median": diff.median(), f"{loss}_q1": diff.quantile(.25),
                      f"{loss}_q3": diff.quantile(.75)})
    return paired, pd.DataFrame(stats).reset_index().sort_values("case_order")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True, help="Archived 200 sensitivity runs")
    parser.add_argument("--mc-root", type=Path, required=True, help="Verified main ensemble; first ten are controls")
    parser.add_argument("--case-registry", type=Path, required=True, help="Pinned EX01 interval registry CSV")
    parser.add_argument("--annual-metrics", type=Path, required=True, help="Pinned EX01 annual archive CSV")
    parser.add_argument("--reference-dir", type=Path, required=True, help="Archived final r1 data directory")
    parser.add_argument("--out-dir", type=Path, help="New directory for verified plot and per-draw CSVs")
    args = parser.parse_args()
    if args.out_dir is not None and (args.out_dir.exists() or args.out_dir.is_symlink()):
        parser.error("--out-dir must not exist")
    try:
        verify(args.mc_root, FINANCE_REFERENCE, 1, 10)
        registry = audited_csv(args.case_registry, REGISTRY_HASH)
        annual_reference = audited_csv(args.annual_metrics, ANNUAL_HASH)
        paired_reference = audited_csv(args.reference_dir / PAIRED, PAIRED_HASH)
        plot_reference = audited_csv(args.reference_dir / PLOT, PLOT_HASH)
        annual = rebuild_annual(args.case_root, args.mc_root, registry)
        delta = compare(annual, annual_reference, ANNUAL_KEYS, ["households", *SUMMED], 1e-5)
        print(f"PASS: 5,460 annual tenure records from 2,730 raw files; max difference {delta:.3g} USD")
        paired, plot = summarize(annual, registry, read_initial_counts())
        paired_delta = compare(paired, paired_reference, PAIR_KEYS, PAIR_STATS, 1e-8)
        print(f"PASS {PAIRED}: {len(paired)} rows, max {paired_delta:.3g} USD/household")
        plot_delta = compare(plot, plot_reference, PLOT_KEYS, PLOT_STATS, 1e-8)
        print(f"PASS {PLOT}: {len(plot)} rows, max {plot_delta:.3g} USD/household")
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True)
            paired[[*PAIR_KEYS, *PAIR_STATS]].to_csv(args.out_dir / PAIRED, index=False, mode="x", lineterminator="\n")
            # Keep the accepted human-readable interval labels, but write the
            # recomputed statistics rather than copying them from the archive.
            labels = plot_reference.drop(columns=PLOT_STATS)
            full = plot.merge(labels, on=PLOT_KEYS, validate="one_to_one")[plot_reference.columns]
            full.to_csv(args.out_dir / PLOT, index=False, mode="x", lineterminator="\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
