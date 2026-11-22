"""Verify and rebuild the flood-sequence sensitivity data for WRR Figure S6."""

import argparse
from collections.abc import Callable
import csv
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PureWindowsPath
from re import fullmatch
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

if __package__:
    from .rebuild_wrr_finance_series import read_initial_counts
    from .verify_wrr_finance import ROOT, SCENARIOS, YEARS
else:
    from rebuild_wrr_finance_series import read_initial_counts
    from verify_wrr_finance import ROOT, SCENARIOS, YEARS


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
PUBLIC_HASHES = ROOT / "data" / "supplementary" / "wrr_s6_finance_reference.csv"
PUBLIC_HASHES_SHA256 = "f057173da49c1f03e5e2613585e7cd60ea75c41fbd326ef5ef2b2e7b6a46f02c"
PUBLIC_PLOT = ROOT / "data" / "figures" / "FigureS6_flood_event_sequence.csv"
PUBLIC_PLOT_SHA256 = "0124047f38d3733c098d953960f6b7280d096dd5fae3c42f44d245ffd960ccf5"
CASES = tuple((order, draw, scenario)
              for order in ORDER_IDS for draw in DRAWS for scenario in SCENARIOS)


def verified_bytes(path: Path, expected: str) -> bytes:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 differs from the audited Figure S6 archive: {path}")
    return payload


def relative_finance_path(order: str, draw: int, scenario: str, year: int) -> Path:
    return (Path("raw") / order / f"draw_{draw:02d}" / scenario / "baseline" / scenario
            / "finance" / f"finance_tract_{year}.csv")


def archived_finance_hashes(suite: Path, references: Path) -> dict[Path, str]:
    registry = pd.read_csv(BytesIO(verified_bytes(suite / REGISTRY, PINNED[REGISTRY])))
    runs = set(CASES)
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
    return signatures


def case_digest(payloads: dict[int, bytes]) -> str:
    """Hash the ordered finance CSVs with checkout-independent line endings."""
    if payloads.keys() != set(YEARS):
        raise ValueError("Expected one annual finance file for each model year")
    digest = hashlib.sha256()
    for year in YEARS:
        name = f"finance_tract_{year}.csv".encode("ascii")
        digest.update(name + b"\0" + payloads[year].replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def read_public_hashes() -> dict[tuple[str, int, str], str]:
    payload = PUBLIC_HASHES.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(payload).hexdigest() != PUBLIC_HASHES_SHA256:
        raise ValueError("Bundled Figure S6 case checksums differ from the revised reference")
    reader = csv.DictReader(payload.decode("utf-8").splitlines())
    if reader.fieldnames != [*RUN_KEYS, "finance_sha256"]:
        raise ValueError("Unexpected Figure S6 reference columns")
    rows = list(reader)
    digests = {(row["order_id"], int(row["draw_id"]), row["scenario"]): row["finance_sha256"]
               for row in rows}
    if (len(rows) != len(CASES) or digests.keys() != set(CASES)
            or any(not value or not fullmatch("[0-9a-f]{64}", value) for value in digests.values())):
        raise ValueError("Expected one valid Figure S6 reference for every order, draw and scenario")
    return digests


def confined_path(root: Path, relative: Path) -> Path:
    path = root / relative
    if not path.is_relative_to(root) or path.resolve() != path:
        raise ValueError(f"Output path does not stay within its suite directory: {path}")
    return path


def read_verified_case(suite: Path, case: tuple[str, int, str], expected: str) -> dict[int, bytes]:
    payloads = {year: confined_path(suite, relative_finance_path(*case, year)).read_bytes()
                for year in YEARS}
    if case_digest(payloads) != expected:
        raise ValueError(f"Revised Figure S6 finance output differs: {case}")
    return payloads


def aggregate_trajectory(
    read_case: Callable[[tuple[str, int, str]], dict[int, bytes]],
) -> pd.DataFrame:
    initial_counts = read_initial_counts()
    cols = [f"{g}_{field}" for g in initial_counts
            for field in ("gross_total_usd", "payout_total_usd", "households")]
    rows = []
    for order, draw, scenario in CASES:
        payloads = read_case((order, draw, scenario))
        for year in YEARS:
            relative = relative_finance_path(order, draw, scenario, year)
            frame = pd.read_csv(BytesIO(payloads[year]), usecols=cols)
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


def rebuild_trajectory(suite: Path, references: Path) -> pd.DataFrame:
    signatures = archived_finance_hashes(suite, references)

    def read_archived_case(case):
        relative = {year: relative_finance_path(*case, year) for year in YEARS}
        return {year: verified_bytes(suite / path, signatures[path]) for year, path in relative.items()}

    return aggregate_trajectory(read_archived_case)


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


def publish_tables(out_dir: Path, trajectory: pd.DataFrame, plot: pd.DataFrame) -> None:
    """Close both staged files before exposing either final CSV, without overwriting."""
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".wrr-s6-", dir=out_dir.parent) as staging:
        staged = Path(staging)
        for name, table in ((TRAJECTORY, trajectory), (PLOT, plot)):
            table.to_csv(staged / name, mode="x", index=False, lineterminator="\n")
        out_dir.mkdir()
        linked = []
        try:
            for name in (TRAJECTORY, PLOT):
                source = staged / name
                dest = out_dir / name
                os.link(source, dest)
                linked.append((source, dest))
        except BaseException:
            for source, dest in reversed(linked):
                if dest.exists() and source.samefile(dest):
                    dest.unlink()
            if not any(out_dir.iterdir()):
                out_dir.rmdir()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True,
                        help="Output from run_wrr_flood_sequence.py, or an archived flood-order experiment")
    parser.add_argument("--reference-dir", type=Path,
                        help="Use the private archived manifest and CSVs instead of the bundled checksums")
    parser.add_argument("--out-dir", type=Path, help="New directory for verified CSV outputs")
    args = parser.parse_args()
    if args.out_dir is not None and (args.out_dir.exists() or args.out_dir.is_symlink()):
        parser.error("--out-dir must not exist")
    try:
        if args.reference_dir is not None:
            trajectory = rebuild_trajectory(args.suite_root, args.reference_dir)
        else:
            suite = args.suite_root.resolve(strict=True)
            hashes = read_public_hashes()
            trajectory = aggregate_trajectory(lambda case: read_verified_case(suite, case, hashes[case]))
        plot = summarize_baseline(trajectory)
        outputs = ((TRAJECTORY, trajectory, RUN_ROW, LOSSES),
                   (PLOT, plot, PLOT_KEYS, PLOT_VALUES))
        for name, table, keys, values in outputs:
            if args.reference_dir is not None:
                payload = verified_bytes(args.reference_dir / name, PINNED[name])
            elif name == PLOT:
                payload = verified_bytes(PUBLIC_PLOT, PUBLIC_PLOT_SHA256)
            else:
                continue
            reference = pd.read_csv(BytesIO(payload), float_precision="round_trip")
            delta = compare(table, reference, keys, values)
            print(f"PASS {name}: {len(table)} rows, maximum absolute difference {delta:.3g} kUSD/household")
        print(f"Verified {len(CASES) * len(YEARS):,} annual finance files across {len(CASES)} sensitivity runs.")
        if args.out_dir is not None:
            publish_tables(args.out_dir, trajectory, plot)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
