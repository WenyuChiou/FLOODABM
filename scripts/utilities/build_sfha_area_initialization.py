"""Build the area-based SFHA attribute and initial flood-insurance status.

Each census tract has one SFHA area share. The same share is applied to
homeowners and renters. Each agent receives an independent, reproducible
uniform draw; ``inside_SFHA`` is one when that draw is below the tract share.
Initial flood insurance is then assigned at fixed rates within each household
group and SFHA status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SFHA_ASSIGNMENT_SEED = 20260817
INITIAL_FI_SEED = 20260818
ROOT = Path(__file__).resolve().parents[2]
INITIAL_FI_RATES = {
    ("owner", 0): 0.03,
    ("owner", 1): 0.30,
    ("renter", 0): 0.02,
    ("renter", 1): 0.08,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_record(path: Path) -> dict[str, str]:
    label = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return {"path": label.as_posix(), "sha256": sha256(path)}


def stable_uniform(base_seed: int, agent_id: object) -> float:
    token = f"{base_seed}|{agent_id}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return integer / float(2**64)


def derived_seed(base_seed: int, group: str, inside_sfha: int) -> int:
    token = f"{base_seed}|{group}|{inside_sfha}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def conventional_round(value: float) -> int:
    return int(math.floor(value + 0.5))


def build(
    households_path: Path,
    area_shares_path: Path,
    output_dir: Path,
    *,
    sfha_seed: int = SFHA_ASSIGNMENT_SEED,
    initial_fi_seed: int = INITIAL_FI_SEED,
    replay: bool = False,
) -> dict:
    households_path = households_path.resolve()
    area_shares_path = area_shares_path.resolve()
    output_dir = output_dir.resolve()
    for path in (households_path, area_shares_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)

    if replay:
        # Keep the published numeric text of the base attributes unchanged.
        # The existing SFHA and FI columns are replaced by seeded assignments.
        households = pd.read_csv(households_path, dtype=str, keep_default_na=False)
    else:
        households = pd.read_csv(households_path, dtype={"tract_geoid": "string"})
    shares = pd.read_csv(area_shares_path, dtype={"tract_geoid": "string"})
    required_household = {"i", "tract_geoid", "group", "has_FI"}
    if replay:
        required_household.update({"sfha_share", "inside_SFHA"})
    required_share = {"tract_geoid", "tract_area_m2", "sfha_area_m2", "sfha_area_share"}
    if missing := required_household - set(households.columns):
        raise ValueError(f"Household file missing columns: {sorted(missing)}")
    if missing := required_share - set(shares.columns):
        raise ValueError(f"Area-share file missing columns: {sorted(missing)}")

    households["tract_geoid"] = households["tract_geoid"].astype(str).str.zfill(11)
    households["group"] = households["group"].astype(str).str.lower()
    shares["tract_geoid"] = shares["tract_geoid"].astype(str).str.zfill(11)
    if not households["group"].isin(["owner", "renter"]).all():
        raise ValueError("Household groups must be owner or renter")
    agent_key = (
        households["tract_geoid"] + "|" + households["group"] + "|"
        + households["i"].astype(str)
    )
    if agent_key.duplicated().any():
        raise ValueError("The tract, group, and agent-ID combination must be unique")
    if shares["tract_geoid"].duplicated().any():
        raise ValueError("Area shares must contain one row per tract")
    shares["sfha_area_share"] = pd.to_numeric(shares["sfha_area_share"], errors="raise")
    if not shares["sfha_area_share"].between(0.0, 1.0).all():
        raise ValueError("SFHA area shares must be in [0, 1]")
    expected_tracts = set(households["tract_geoid"].unique())
    observed_tracts = set(shares["tract_geoid"].unique())
    if expected_tracts != observed_tracts:
        raise ValueError(
            "Household and area-share tract sets differ: "
            f"missing={sorted(expected_tracts-observed_tracts)}, "
            f"extra={sorted(observed_tracts-expected_tracts)}"
        )

    share_table = shares.rename(columns={"sfha_area_share": "sfha_share"}).copy()
    share_table["share_status"] = "SFHA_area_fraction"
    share_lookup = share_table.set_index("tract_geoid")["sfha_share"]
    households["sfha_share"] = households["tract_geoid"].map(share_lookup)
    uniforms = agent_key.map(lambda value: stable_uniform(sfha_seed, value))
    households["inside_SFHA"] = (uniforms < households["sfha_share"]).astype("int8")

    households["has_FI"] = np.int8(0)
    fi_summary: list[dict] = []
    for key, rate in INITIAL_FI_RATES.items():
        group, inside_sfha = key
        mask = households["group"].eq(group) & households["inside_SFHA"].eq(inside_sfha)
        indices = households.index[mask].to_numpy()
        target = conventional_round(len(indices) * rate)
        rng = np.random.default_rng(derived_seed(initial_fi_seed, group, inside_sfha))
        if target:
            selected = rng.choice(indices, size=target, replace=False)
            households.loc[selected, "has_FI"] = np.int8(1)
        fi_summary.append(
            {
                "group": group,
                "inside_SFHA": inside_sfha,
                "households": int(len(indices)),
                "target_rate": rate,
                "initial_FI": int(target),
                "observed_rate": target / len(indices) if len(indices) else 0.0,
            }
        )

    assignment_summary = (
        households.groupby(["tract_geoid", "group"], as_index=False)
        .agg(
            households=("i", "size"),
            assigned_inside_SFHA=("inside_SFHA", "sum"),
            input_sfha_share=("sfha_share", "first"),
        )
    )
    assignment_summary["assigned_share"] = (
        assignment_summary["assigned_inside_SFHA"] / assignment_summary["households"]
    )

    household_output = output_dir / "households_for_abm_sfha.csv"
    share_output = output_dir / "sfha_shares.csv"
    assignment_output = output_dir / "sfha_assignment_summary.csv"
    fi_output = output_dir / "initial_fi_summary.csv"
    manifest_output = output_dir / "sfha_assignment_manifest.json"
    households.to_csv(household_output, index=False)
    share_table[
        ["tract_geoid", "tract_area_m2", "sfha_area_m2", "sfha_share", "share_status"]
    ].to_csv(share_output, index=False)
    assignment_summary.to_csv(assignment_output, index=False)
    pd.DataFrame(fi_summary).to_csv(fi_output, index=False)

    manifest = {
        "schema_version": "2.1",
        "design": "tract SFHA area share applied equally to homeowners and renters",
        "sfha_assignment_method": (
            "inside_SFHA = 1 when the stable uniform draw for tract|group|agent-ID "
            "is below the tract SFHA area share"
        ),
        "sfha_assignment_seed": sfha_seed,
        "initial_FI_assignment_seed": initial_fi_seed,
        "initial_FI_rates": {
            f"{group}|inside_SFHA={inside}": rate
            for (group, inside), rate in INITIAL_FI_RATES.items()
        },
        "household_input_mode": "replay" if replay else "unassigned",
        "source_households": source_record(households_path),
        "source_area_shares": source_record(area_shares_path),
        "outputs": {
            path.name: sha256(path)
            for path in (household_output, share_output, assignment_output, fi_output)
        },
        "n_households": int(len(households)),
        "n_tracts": int(households["tract_geoid"].nunique()),
        "inside_SFHA": int(households["inside_SFHA"].sum()),
        "initial_FI": int(households["has_FI"].sum()),
        "hash_format": "SHA-256 of file bytes with CRLF normalized to LF",
        "fi_strata": fi_summary,
    }
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--households", type=Path, required=True)
    parser.add_argument("--area-shares", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sfha-seed", type=int, default=SFHA_ASSIGNMENT_SEED)
    parser.add_argument("--initial-fi-seed", type=int, default=INITIAL_FI_SEED)
    parser.add_argument(
        "--replay", action="store_true",
        help="Recalculate the assignments from the published household CSV without changing its base attributes.",
    )
    args = parser.parse_args()
    build(
        args.households,
        args.area_shares,
        args.output_dir,
        sfha_seed=args.sfha_seed,
        initial_fi_seed=args.initial_fi_seed,
        replay=args.replay,
    )


if __name__ == "__main__":
    main()
