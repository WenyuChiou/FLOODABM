"""Validate the committed SFHA-aware household initialization inputs.

This check is intentionally independent of the simulation loop. It verifies
that SFHA assignment changes only the new attribute, preserves the original
initial FI flags, and contains one valid share row for each study tract and
tenure group.
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
    key = ["i", "tract_geoid", "group"]
    left = old.sort_values("i").reset_index(drop=True)
    right = new.sort_values("i").reset_index(drop=True)
    for col in ["tract_geoid", "group", "has_FI"]:
        if not left[col].astype(str).equals(right[col].astype(str)):
            raise AssertionError(f"SFHA assignment changed protected column: {col}")
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
    print(json.dumps({
        "status": "PASS",
        "n_households": int(len(new)),
        "n_tracts": int(new["tract_geoid"].nunique()),
        "inside_SFHA": int(new["inside_SFHA"].sum()),
        "initial_has_FI": int(new["has_FI"].sum()),
        "fallback": "34027040302/renter=0",
    }, indent=2))


if __name__ == "__main__":
    main()
