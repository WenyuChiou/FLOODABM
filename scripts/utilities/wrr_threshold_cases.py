"""Case identifiers, finance checksums and paths shared by revised S4 and S5."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from re import fullmatch

if __package__:
    from .rebuild_wrr_sequence_series import ROOT, YEARS, case_digest, confined_path
    from .verify_wrr_finance import reference_hashes
else:
    from rebuild_wrr_sequence_series import ROOT, YEARS, case_digest, confined_path
    from verify_wrr_finance import reference_hashes


DRAW_IDS = tuple(range(1, 11))
S4_COMPONENTS = {"FI_homeowner_outside_SFHA": "owner", "FI_renter": "renter",
                 "EH": "owner", "BP": "owner", "RL": "renter"}
S4_VARIATIONS = ("lower", "higher", "narrower", "wider")
S5_TENURE = {"homeowner": "owner", "renter": "renter"}
S5_THRESHOLDS = tuple(range(20, 100, 10))
CASE_IDS = {
    4: tuple(f"EX01_{component}_{variant}" for component in S4_COMPONENTS for variant in S4_VARIATIONS),
    5: tuple(f"EX02_{tenure}_{code:03d}" for tenure in S5_TENURE for code in S5_THRESHOLDS),
}
REFERENCE_SHA256 = {
    4: "48b4e003f5396d132678142fc68abbf57b1067e99954a460591f221e366d3b56",
    5: "3d86ee8a3ae6c348fed40de538b6b0e9f9eecf5b64113952d82e9ebcc94f012f",
}
PLOT_REFERENCES = {
    4: ("FigureS4_action_threshold_intervals.csv",
        "c1f14f4ff886ebab613be57690fd92cfb37d1fe1c3f58882d684cbc49acfaf64"),
    5: ("FigureS5_gul_ratio_threshold.csv",
        "39b2a0a8a1a5f8ced2cbcd6b3dc25caea1d975e86e209c0e394f34c36c612cf9"),
}
ControlPayloads = dict[int, dict[int, bytes]]


def case_output_root(number: int, case_id: str, draw: int) -> Path:
    if case_id not in CASE_IDS[number] or draw not in DRAW_IDS:
        raise ValueError(f"Not a revised Figure S{number} case/draw: {case_id!r} / {draw!r}")
    if number == 4:
        return Path(case_id) / f"run_{draw:02d}" / "model_output"
    suffix = case_id.removeprefix("EX02_")
    return Path(f"run_{draw:02d}_{suffix}") / "model_output"


def case_model_args(number: int, case_id: str) -> tuple[str, ...]:
    if case_id not in CASE_IDS[number]:
        raise ValueError(f"Not a revised Figure S{number} case: {case_id!r}")
    if number == 4:
        return "--wrr-s4-case", case_id
    _, tenure, code = case_id.split("_")
    varied = str(int(code) / 100)
    return ("--thr-owner", varied if tenure == "homeowner" else "0.1",
            "--thr-renter", varied if tenure == "renter" else "0.1")


def load_hashes(number: int) -> dict[tuple[str, int], str]:
    path = ROOT / "data" / "supplementary" / f"wrr_s{number}_finance_reference.csv"
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(payload).hexdigest() != REFERENCE_SHA256[number]:
        raise ValueError(f"Bundled Figure S{number} checksums differ from the revised reference")
    reader = csv.DictReader(payload.decode("utf-8").splitlines())
    if reader.fieldnames != ["case_id", "run_number", "finance_sha256"]:
        raise ValueError(f"Unexpected Figure S{number} reference columns")
    rows = list(reader)
    actual = {(row["case_id"], int(row["run_number"])): row["finance_sha256"] for row in rows}
    expected = {(case, draw) for case in CASE_IDS[number] for draw in DRAW_IDS}
    if len(rows) != len(expected) or actual.keys() != expected or not all(
        value and fullmatch("[0-9a-f]{64}", value) for value in actual.values()
    ):
        raise ValueError(f"Expected one checksum per revised Figure S{number} case and draw")
    return actual


def verified_finance(root: Path, number: int, case_id: str, draw: int, expected: str) -> dict[int, bytes]:
    finance = case_output_root(number, case_id, draw) / "baseline" / "baseline" / "finance"
    data = {year: confined_path(root, finance / f"finance_tract_{year}.csv").read_bytes() for year in YEARS}
    if case_digest(data) != expected:
        raise ValueError(f"Revised Figure S{number} finance differs: {case_id}, draw {draw}")
    return data


def bundled_plot(number: int) -> bytes:
    name, expected = PLOT_REFERENCES[number]
    payload = (ROOT / "data" / "figures" / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Bundled Figure S{number} plot differs from the revised reference")
    return payload


def snapshot_controls(root: Path, reference: Path) -> ControlPayloads:
    """Verify and retain exactly the control bytes that will be aggregated."""
    root = root.resolve(strict=True)
    expected = reference_hashes(reference)
    snapshots = {}
    for draw in DRAW_IDS:
        finance = Path(f"run_{draw:02d}") / "baseline" / "baseline" / "finance"
        data = {year: confined_path(root, finance / f"finance_tract_{year}.csv").read_bytes()
                for year in YEARS}
        if case_digest(data) != expected[draw, "baseline"]:
            raise ValueError(f"Revised main-ensemble baseline control differs: draw {draw}")
        snapshots[draw] = data
    return snapshots
