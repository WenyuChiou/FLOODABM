"""Verify and rebuild the flood-sequence sensitivity data for WRR Figure S6."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

if __package__:
    from .rebuild_wrr_finance_series import read_initial_counts
    from .verify_wrr_finance import SCENARIOS, YEARS
else:
    from rebuild_wrr_finance_series import read_initial_counts
    from verify_wrr_finance import SCENARIOS, YEARS


REGISTRY = "processed/run_registry.csv"
RAW_HASHES = "raw_finance_hashes.json"
TRAJECTORY = "fixed_initial_trajectory_by_run.csv"
PLOT = "Figure_R3_fixed_initial_plot_data.csv"
PINNED = {
    REGISTRY: "615c191d058ea18e032cefe1a9748b55d2fb328244e9327ecfcd17f3a259361f",
    RAW_HASHES: "85364dd599374cbbea424e86a286d0b7ec33d8b306638e91d04f46ce54aafd08",
    TRAJECTORY: "ea5beec8470f9cf6fc889b945f1a688464e61dac7561a39203de7d827854cd05",
    PLOT: "2a174de0acb55f5527b8f77e3ad6d61b98b400dc8b97c448471799c4ac077b61",
}
ORDER_IDS = ("observed", *(f"permuted_{i:02d}" for i in range(1, 11)))
DRAWS = range(1, 11)
RUN_KEYS = ["order_id", "draw_id", "scenario"]
RUN_ROW = [*RUN_KEYS, "year", "group"]
LOSSES = ["GUL", "actual_loss"]
PLOT_KEYS = ["year", "group", "metric", "sequence"]
PLOT_VALUES = ["center", "lower", "upper"]


def verified_bytes(path: Path, expected: str) -> bytes:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 differs from the audited Figure S6 archive: {path}")
    return payload


def relative_finance_path(order: str, draw: int, scenario: str, year: int) -> Path:
    return (Path("raw") / order / f"draw_{draw:02d}" / scenario / "baseline" / scenario
            / "finance" / f"finance_tract_{year}.csv")


def rebuild_trajectory(suite: Path, references: Path) -> pd.DataFrame:
    registry = pd.read_csv(BytesIO(verified_bytes(suite / REGISTRY, PINNED[REGISTRY])))
    runs = {(order, draw, scenario) for order in ORDER_IDS for draw in DRAWS for scenario in SCENARIOS}
    if (len(registry) != len(runs) or not registry["status"].eq("PASS").all()
            or set(registry[RUN_KEYS].itertuples(index=False, name=None)) != runs):
        raise ValueError("Expected every observed/shuffled draw and both scenarios exactly once")
    hashes = json.loads(verified_bytes(references / RAW_HASHES, PINNED[RAW_HASHES]))
    # Map only the known archive-relative filenames, never open the archived
    # host's absolute paths. Their manifest is a pinned reference.
    signatures = {Path(*PureWindowsPath(row["path"]).parts[-8:]): row["sha256"] for row in hashes}
    required = {relative_finance_path(*run, year) for run in runs for year in YEARS}
    if len(hashes) != len(required) or signatures.keys() != required:
        raise ValueError("Archive finance hash manifest differs from the expected file grid")
    initial_counts = read_initial_counts()
    cols = [f"{g}_{field}" for g in initial_counts
            for field in ("gross_total_usd", "payout_total_usd", "households")]
    rows = []
    for order, draw, scenario in sorted(runs):
        for year in YEARS:
            relative = relative_finance_path(order, draw, scenario, year)
            frame = pd.read_csv(BytesIO(verified_bytes(suite / relative, signatures[relative])),
                                usecols=cols)
            for group, count in initial_counts.items():
                gross = frame[f"{group}_gross_total_usd"].sum()
                actual = gross - frame[f"{group}_payout_total_usd"].sum()
                if year == YEARS.start and frame[f"{group}_households"].sum() != count:
                    raise ValueError(f"Unexpected initial tenure population: {relative}")
                rows.append((order, draw, scenario, year, group, gross / count / 1000,
                             actual / count / 1000))
    result = pd.DataFrame(rows, columns=[*RUN_ROW, *LOSSES]).sort_values(RUN_ROW)
    result[LOSSES] = result.groupby([*RUN_KEYS, "group"])[LOSSES].cumsum()
    return result


def summarize_baseline(trajectory: pd.DataFrame) -> pd.DataFrame:
    adapted = trajectory.loc[trajectory["scenario"].eq("baseline")]
    long = adapted.melt(id_vars=["order_id", "draw_id", "year", "group"], value_vars=LOSSES,
                        var_name="metric", value_name="value")
    keys = PLOT_KEYS[:-1]
    observed = long.loc[long["order_id"].eq("observed")].groupby(keys)["value"].agg(
        center="median", lower=lambda values: values.quantile(.25),
        upper=lambda values: values.quantile(.75),
    ).reset_index().assign(sequence="Observed sequence")
    orders = long.loc[long["order_id"].isin(ORDER_IDS[1:])]
    medians = orders.groupby(["order_id", *keys])["value"].median().reset_index()
    shuffled = medians.groupby(keys)["value"].agg(center="median", lower="min", upper="max")
    return pd.concat([observed, shuffled.reset_index().assign(sequence="Shuffled sequences")],
                     ignore_index=True)


def compare(table: pd.DataFrame, reference: pd.DataFrame, keys: list[str], values: list[str]) -> float:
    a = table.set_index(keys, verify_integrity=True).sort_index()
    b = reference.set_index(keys, verify_integrity=True).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Rebuilt sequence keys differ from the accepted archive")
    delta = (a[values] - b[values]).abs().to_numpy()
    if not np.isfinite(delta).all() or delta.max() > 1e-9:
        raise ValueError("Rebuilt sequence quantities differ from the accepted archive")
    return float(delta.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True,
                        help="Archived flood-order experiment with raw/ and processed/")
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Archived r3 directory containing the audited manifest and CSVs")
    parser.add_argument("--out-dir", type=Path, help="New directory for verified CSV outputs")
    args = parser.parse_args()
    if args.out_dir is not None and (args.out_dir.exists() or args.out_dir.is_symlink()):
        parser.error("--out-dir must not exist")
    try:
        trajectory = rebuild_trajectory(args.suite_root, args.reference_dir)
        plot = summarize_baseline(trajectory)
        outputs = ((TRAJECTORY, trajectory, RUN_ROW, LOSSES),
                   (PLOT, plot, PLOT_KEYS, PLOT_VALUES))
        for name, table, keys, values in outputs:
            reference = pd.read_csv(BytesIO(verified_bytes(args.reference_dir / name, PINNED[name])),
                                    float_precision="round_trip")
            delta = compare(table, reference, keys, values)
            print(f"PASS {name}: {len(table)} rows, maximum absolute difference {delta:.3g} kUSD/household")
        print("Verified all 2,860 annual finance hashes across 220 sensitivity runs.")
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True)
            for name, table, _, _ in outputs:
                table.to_csv(args.out_dir / name, mode="x", index=False, lineterminator="\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
