"""Validate the area-based SFHA and initial flood-insurance inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "config" / "households_for_abm.csv"
NEW = ROOT / "config" / "households_for_abm_sfha.csv"
SHARES = ROOT / "config" / "sfha_shares.csv"
MANIFEST = ROOT / "config" / "sfha_assignment_manifest.json"
CONFIG = ROOT / "config" / "abm_params.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def stable_uniform(base_seed: int, agent_key: str) -> float:
    token = f"{base_seed}|{agent_key}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return integer / float(2**64)


def main() -> None:
    old = pd.read_csv(OLD, dtype={"tract_geoid": str})
    new = pd.read_csv(NEW, dtype={"tract_geoid": str})
    shares = pd.read_csv(SHARES, dtype={"tract_geoid": str})
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}

    if len(old) != len(new) or set(old["i"]) != set(new["i"]):
        raise AssertionError("SFHA initialization changed household rows or IDs")
    left = old.sort_values("i").reset_index(drop=True)
    right = new.sort_values("i").reset_index(drop=True)
    for column in [name for name in left.columns if name != "has_FI"]:
        if not left[column].astype(str).equals(right[column].astype(str)):
            raise AssertionError(f"SFHA initialization changed protected column: {column}")
    for column in ("inside_SFHA", "has_FI"):
        if column not in right or not right[column].isin([0, 1]).all():
            raise AssertionError(f"{column} must contain only 0/1 values")

    required_share_columns = {
        "tract_geoid", "tract_area_m2", "sfha_area_m2", "sfha_share", "share_status"
    }
    if required_share_columns - set(shares.columns):
        raise AssertionError("SFHA area-share file is missing required columns")
    if len(shares) != 27 or shares["tract_geoid"].duplicated().any():
        raise AssertionError("SFHA shares must contain exactly one row for each of 27 tracts")
    if not shares["sfha_share"].between(0.0, 1.0).all():
        raise AssertionError("SFHA shares must be in [0, 1]")
    if set(shares["share_status"]) != {"SFHA_area_fraction"}:
        raise AssertionError("SFHA share provenance label is incorrect")

    share_lookup = shares.set_index("tract_geoid")["sfha_share"]
    expected_share = right["tract_geoid"].map(share_lookup)
    if expected_share.isna().any() or not expected_share.equals(right["sfha_share"]):
        raise AssertionError("Agent sfha_share does not match its tract area share")
    seed = int(manifest["sfha_assignment_seed"])
    expected_inside = right.apply(
        lambda row: int(
            stable_uniform(
                seed, f"{row['tract_geoid']}|{row['group']}|{row['i']}"
            ) < float(row["sfha_share"])
        ),
        axis=1,
    )
    if not expected_inside.equals(right["inside_SFHA"].astype(int)):
        raise AssertionError("inside_SFHA is not the recorded Bernoulli assignment")

    source_area = ROOT / "config" / "sfha_area_share_by_tract.csv"
    for label, path in (("source_households", OLD), ("source_area_shares", source_area)):
        if manifest[label]["path"] != path.relative_to(ROOT).as_posix():
            raise AssertionError(f"Unexpected manifest source: {label}")
        if manifest[label]["sha256"] != sha256(path):
            raise AssertionError(f"Manifest source hash differs: {path.name}")
    expected = {NEW.name, SHARES.name, "sfha_assignment_summary.csv", "initial_fi_summary.csv"}
    if set(manifest["outputs"]) != expected:
        raise AssertionError("Incomplete manifest outputs")
    for name in expected:
        if manifest["outputs"][name] != sha256(ROOT / "config" / name):
            raise AssertionError(f"Manifest output hash differs: {name}")
    rates = {
        tuple(key.replace("inside_SFHA=", "").split("|")): value
        for key, value in manifest["initial_FI_rates"].items()
    }
    rates = {(group, int(inside)): float(rate) for (group, inside), rate in rates.items()}
    sizes = right.groupby(["group", "inside_SFHA"], observed=True).size().to_dict()
    expected_counts = {
        key: int((sizes[key] * rate + 0.5) // 1) for key, rate in rates.items()
    }
    observed_counts = (
        right.groupby(["group", "inside_SFHA"], observed=True)["has_FI"]
        .sum().astype(int).to_dict()
    )
    if observed_counts != expected_counts:
        raise AssertionError(f"Initial FI counts differ from the specified rates: {observed_counts}")

    intervals = {
        "homeowner_inside": list(config["sfha_initialization"]["fi_draw_bounds_inside"]),
        "homeowner_outside": list(config["draw_bounds"]["FI"]),
        "renter": list(config["draw_bounds"]["FI_renter"]),
    }
    if intervals != {
        "homeowner_inside": [0.0, 0.1],
        "homeowner_outside": [0.35, 0.55],
        "renter": [0.7, 0.9],
    }:
        raise AssertionError("FI decision intervals do not match the revised design")

    by_group = right.groupby("group", observed=True)["inside_SFHA"].agg(["sum", "count"])
    by_group["share"] = by_group["sum"] / by_group["count"]
    print(json.dumps({
        "status": "PASS",
        "n_households": int(len(right)),
        "n_tracts": int(right["tract_geoid"].nunique()),
        "inside_SFHA": int(right["inside_SFHA"].sum()),
        "inside_SFHA_by_group": {
            group: {"count": int(row["sum"]), "share": float(row["share"])}
            for group, row in by_group.iterrows()
        },
        "initial_has_FI": int(right["has_FI"].sum()),
        "initial_FI_counts": {
            f"{group}_{'inside' if inside else 'outside'}_SFHA": count
            for (group, inside), count in observed_counts.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
