"""Validate the committed SFHA-aware household initialization inputs.

This check is intentionally independent of the simulation loop. It verifies
that SFHA assignment preserves the original household attributes, except for
the revised initial FI flag, and contains one valid share row for each study
tract and homeowner/renter group.
"""
from __future__ import annotations

from pathlib import Path
import json
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "config" / "households_for_abm.csv"
NEW = ROOT / "config" / "households_for_abm_sfha.csv"
SHARES = ROOT / "config" / "sfha_shares.csv"
MANIFEST = ROOT / "config" / "sfha_assignment_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    old = pd.read_csv(OLD, dtype={"tract_geoid": str})
    new = pd.read_csv(NEW, dtype={"tract_geoid": str})
    shares = pd.read_csv(SHARES, dtype={"tract_geoid": str, "_group": str})
    if len(old) != len(new) or set(old["i"]) != set(new["i"]):
        raise AssertionError("SFHA input changed household rows or IDs")
    left = old.sort_values("i").reset_index(drop=True)
    right = new.sort_values("i").reset_index(drop=True)
    protected_columns = [column for column in left.columns if column != "has_FI"]
    for col in protected_columns:
        if not left[col].astype(str).equals(right[col].astype(str)):
            raise AssertionError(f"SFHA assignment changed protected column: {col}")
    if not right["has_FI"].isin([0, 1]).all():
        raise AssertionError("has_FI must be a binary household attribute")
    if "inside_SFHA" not in right.columns or not right["inside_SFHA"].isin([0, 1]).all():
        raise AssertionError("inside_SFHA must be a binary household attribute")
    if len(shares) != 54 or shares.duplicated(["tract_geoid", "_group"]).any():
        raise AssertionError("SFHA shares must contain exactly 27 tracts × 2 tenure groups")
    if not shares["sfha_share"].between(0.0, 1.0).all():
        raise AssertionError("SFHA shares must be in [0, 1]")
    fallback = shares[shares["share_status"].eq("no_supported_NSI_record_assumed_zero")]
    if len(fallback) != 1 or tuple(fallback[["tract_geoid", "_group"]].iloc[0]) != ("34027040302", "renter"):
        raise AssertionError("The documented renter fallback is missing or changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["household_path"]["sha256"].lower()
    if expected != sha256(NEW):
        raise AssertionError("Assignment manifest does not bind the committed SFHA household file")
    expected_counts = {
        ("owner", 0): manifest["initial_FI"]["counts"]["owner_outside_SFHA"],
        ("owner", 1): manifest["initial_FI"]["counts"]["owner_inside_SFHA"],
        ("renter", 0): manifest["initial_FI"]["counts"]["renter_outside_SFHA"],
        ("renter", 1): manifest["initial_FI"]["counts"]["renter_inside_SFHA"],
    }
    rates = manifest["initial_FI"]["rates_pct"]
    expected_rates = {
        ("owner", 0): rates["owner"]["outside_SFHA"],
        ("owner", 1): rates["owner"]["inside_SFHA"],
        ("renter", 0): rates["renter"]["outside_SFHA"],
        ("renter", 1): rates["renter"]["inside_SFHA"],
    }
    stratum_sizes = right.groupby(["group", "inside_SFHA"], observed=True).size().to_dict()
    derived_counts = {
        key: int(round(stratum_sizes[key] * rate / 100.0))
        for key, rate in expected_rates.items()
    }
    if expected_counts != derived_counts:
        raise AssertionError(
            f"Manifest FI counts do not match its rates: {derived_counts}"
        )
    observed_counts = (
        right.groupby(["group", "inside_SFHA"], observed=True)["has_FI"]
        .sum()
        .astype(int)
        .to_dict()
    )
    if observed_counts != expected_counts:
        raise AssertionError(
            f"Initial FI counts do not match the manifest: {observed_counts}"
        )
    if int(right["has_FI"].sum()) != int(manifest["initial_FI"]["counts"]["total"]):
        raise AssertionError("Initial FI total does not match the manifest")
    print(json.dumps({
        "status": "PASS",
        "n_households": int(len(new)),
        "n_tracts": int(new["tract_geoid"].nunique()),
        "inside_SFHA": int(new["inside_SFHA"].sum()),
        "initial_has_FI": int(new["has_FI"].sum()),
        "initial_FI_counts": {
            f"{group}_{'inside' if inside else 'outside'}_SFHA": count
            for (group, inside), count in observed_counts.items()
        },
        "fallback": "34027040302/renter=0",
    }, indent=2))


if __name__ == "__main__":
    main()
