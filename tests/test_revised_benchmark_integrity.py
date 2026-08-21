from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from main import (
    _prepare_initial_fi,
    _refresh_sfha_after_relocation,
    _resolve_simulation_seeds,
    sample_event_depths,
    validate_depth_distribution_alignment,
)
from modules.actions.bayes_fast_predictors import build_fast_predictors
from modules.actions import vuln_for_tp
from scripts.sensitivity.run_sensitivity_analysis import require_legacy_initial_fi
from scripts.utilities.generate_household_psych import generate_initial_conditions
from scripts.utilities.run_mc100_local import (
    _sha256_files,
    build_design,
    decision_seed,
    posterior_index,
    validate_benchmark_config,
)
from utils.flood_prone import flood_prone_flags


ROOT = Path(__file__).resolve().parents[1]


def test_posterior_draw_uses_tracked_nonconstant_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOODABM_POSTERIOR_IDX", "864")
    owner, renter = build_fast_predictors(ROOT / "models" / "baseline")
    inputs = {
        "TP": np.array([0.2, 0.8]),
        "CP": np.array([0.3, 0.7]),
        "SP": np.array([0.4, 0.6]),
    }
    owner_probabilities = owner(**inputs)
    renter_probabilities = renter(**inputs)
    for action in ("FI", "EH", "BP"):
        assert np.isfinite(owner_probabilities[action]).all()
        assert not np.allclose(owner_probabilities[action], 0.5)
    for action in ("FI", "RL"):
        assert np.isfinite(renter_probabilities[action]).all()
        assert not np.allclose(renter_probabilities[action], 0.5)


def test_invalid_posterior_index_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOODABM_POSTERIOR_IDX", "4800")
    with pytest.raises(RuntimeError, match="outside"):
        build_fast_predictors(ROOT / "models" / "baseline")


def test_depth_distribution_contract_and_deterministic_sampling() -> None:
    path = ROOT / "config" / "depth_distribution_by_tract_year.csv"
    frame = pd.read_csv(path, dtype={"tract_geoid": str})
    assert len(frame) == 351
    assert frame["year"].nunique() == 13
    assert frame["tract_geoid"].nunique() == 27
    assert not frame[["year", "tract_geoid"]].duplicated().any()

    row = frame.iloc[0]
    depths = np.asarray(json.loads(row["depth_values_m"]), dtype=float)
    assert len(depths) == int(row["n_cells"])
    assert np.isfinite(depths).all()
    assert (depths >= 0).all()
    assert np.isclose(depths.mean(), float(row["mean_depth_m"]), rtol=0, atol=1e-12)

    state = pd.DataFrame({"i": [1, 2, 3], "tract_geoid": ["a", "a", "a"]})
    distributions = {(2011, "a"): np.array([0.0, 0.5, 1.0])}
    first = sample_event_depths(state, 2011, distributions, np.random.RandomState(7))
    second = sample_event_depths(state, 2011, distributions, np.random.RandomState(7))
    pd.testing.assert_frame_equal(first, second)


def test_depth_distribution_must_match_mean_depth_hazard() -> None:
    means = pd.DataFrame(
        {"year": [2011], "tract_geoid": ["a"], "depth_m": [0.5]}
    )
    validate_depth_distribution_alignment(
        means, {(2011, "a"): np.array([0.0, 1.0])}
    )
    with pytest.raises(ValueError, match="different hazards"):
        validate_depth_distribution_alignment(
            means, {(2011, "a"): np.array([1.0, 2.0])}
        )
    with pytest.raises(ValueError, match="matching tract-year coverage"):
        validate_depth_distribution_alignment(
            means, {(2012, "a"): np.array([0.0, 1.0])}
        )


def test_event_depth_override_is_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    state = pd.DataFrame({"i": [1, 2], "group": ["owner", "renter"]})
    override = pd.DataFrame({"i": [1, 2], "event_depth_m": [0.25, 1.50]})

    monkeypatch.setattr(vuln_for_tp, "depths_for_year", lambda *_: pd.DataFrame())
    monkeypatch.setattr(
        vuln_for_tp,
        "build_cat_input",
        lambda df, *_args, **_kwargs: pd.DataFrame({"i": df["i"], "depth_m": 0.0}),
    )

    def fake_losses(cat_in: pd.DataFrame, *_args, **_kwargs) -> pd.DataFrame:
        depth = cat_in["depth_m"].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "i": cat_in["i"],
                "gross_structure_loss": depth,
                "gross_contents_loss": depth * 2,
                "gross_total": depth * 3,
            }
        )

    monkeypatch.setattr(vuln_for_tp, "compute_losses_quick", fake_losses)
    result = vuln_for_tp._attach_hh_flood_damage(
        state,
        2011,
        pd.DataFrame(),
        ROOT / "modules",
        renters_have_structure=True,
        event_depth_m=override,
    )
    assert result["event_depth_m"].tolist() == [0.25, 1.50]
    assert result["gross_total_kUSD"].tolist() == [0.75, 4.50]


def test_missing_destination_sfha_share_fails() -> None:
    before = pd.DataFrame({"i": [1], "tract_geoid": ["a"], "inside_SFHA": [0]})
    after = pd.DataFrame(
        {"i": [1], "tract_geoid": ["b"], "group": ["renter"], "inside_SFHA": [0]}
    )
    with pytest.raises(KeyError):
        _refresh_sfha_after_relocation(before, after, {}, seed=1, year=2012)


def test_sfha_benchmark_requires_preassigned_initial_fi() -> None:
    state = pd.DataFrame(
        {"i": [1], "identity": ["owner"], "tract_geoid": ["34013021300"]}
    )
    with pytest.raises(ValueError, match="requires preassigned has_FI"):
        _prepare_initial_fi(state, {}, sfha_enabled=True, seed=42)


def test_preassigned_initial_fi_is_strict_and_preserved() -> None:
    state = pd.DataFrame({"i": [1, 2], "has_FI": [1, 0]})
    result = _prepare_initial_fi(state, {}, sfha_enabled=True, seed=42)
    assert result["has_FI"].tolist() == [1, 0]
    assert str(result["has_FI"].dtype) == "int8"

    invalid = pd.DataFrame({"i": [1, 2], "has_FI": [1, 2]})
    with pytest.raises(ValueError, match="only 0/1"):
        _prepare_initial_fi(invalid, {}, sfha_enabled=True, seed=42)


def test_sfha_household_generator_preserves_initial_fi(tmp_path: Path) -> None:
    csv_path = tmp_path / "households.csv"
    yaml_path = tmp_path / "config.yaml"
    original = pd.DataFrame(
        {
            "i": [1, 2],
            "tract_geoid": ["34013021300", "34013021300"],
            "group": ["owner", "renter"],
            "inside_SFHA": [1, 0],
            "has_FI": [1, 0],
        }
    )
    original.to_csv(csv_path, index=False)
    yaml_path.write_text("sfha_initialization:\n  enabled: true\n", encoding="utf-8")

    result = generate_initial_conditions(csv_path, yaml_path=yaml_path, seed=2025)

    assert result["inside_SFHA"].tolist() == [1, 0]
    assert result["has_FI"].tolist() == [1, 0]


def test_legacy_fi_sensitivity_rejects_sfha_benchmark() -> None:
    with pytest.raises(RuntimeError, match="cannot run with the SFHA-aware benchmark"):
        require_legacy_initial_fi("SA2")


def test_non_sfha_initial_fi_requires_explicit_legacy_rates() -> None:
    state = pd.DataFrame(
        {"i": [1], "identity": ["owner"], "tract_geoid": ["34013021300"]}
    )
    with pytest.raises(ValueError, match="refusing to create an all-zero"):
        _prepare_initial_fi(state, {}, sfha_enabled=False, seed=42)


def test_shared_flood_prone_classification_uses_seven_year_rule(
    tmp_path: Path,
) -> None:
    rows = []
    for year in range(2011, 2024):
        rows.extend(
            [
                {"year": year, "tract_geoid": "a", "depth_m": 0.1 if year <= 2017 else 0.0},
                {"year": year, "tract_geoid": "b", "depth_m": 0.1 if year <= 2016 else 0.0},
            ]
        )
    flood_path = tmp_path / "flood_years_by_tract.csv"
    pd.DataFrame(rows).to_csv(flood_path, index=False)

    flags = flood_prone_flags(flood_path, ["a", "b"])

    assert flags.to_dict() == {"a": 1, "b": 0}


def test_mc50_design_is_fixed_and_matches_active_benchmark() -> None:
    assert decision_seed(1) == 90001
    assert decision_seed(50) == 91814
    assert posterior_index(1) == 0
    assert posterior_index(50) == 4704
    assert _resolve_simulation_seeds(90334, None) == (90334, 90334)
    assert _resolve_simulation_seeds(90334, 90001) == (90334, 90001)
    validate_benchmark_config()
    design = build_design()
    assert design["fingerprints"]["code"]["n_files"] > 1
    assert design["fingerprints"]["posterior_models"]["n_files"] >= 5
    assert set(design["fingerprints"]["inputs"]) >= {
        "config",
        "households",
        "mean_depths",
        "depth_distributions",
    }


def test_mc_fingerprint_accepts_external_inputs(tmp_path: Path) -> None:
    external = tmp_path / "custom_depths.csv"
    external.write_text("year,tract_geoid,depth_values_m\n", encoding="utf-8")
    result = _sha256_files([external])
    assert result["n_files"] == 1
    assert len(result["sha256"]) == 64
