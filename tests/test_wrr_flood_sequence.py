import numpy as np
import pandas as pd
import pytest

import main as runner
from main import main, run_single_scenario
from modules.wrr_flood_sequence import (
    EVENT_ORDERS,
    YEARS,
    depth_sampling_seed,
    relabel_depth_inputs,
)


def test_saved_orders_match_the_archived_permutation_design():
    rng = np.random.default_rng(20260823)
    generated = [YEARS]
    while len(generated) < 11:
        order = tuple(int(year) for year in rng.permutation(YEARS))
        if order not in generated:
            generated.append(order)
    assert list(EVENT_ORDERS) == ["observed", *(f"permuted_{i:02}" for i in range(1, 11))]
    assert list(EVENT_ORDERS.values()) == generated


@pytest.fixture
def inputs():
    table = pd.DataFrame({"year": YEARS, "tract_geoid": "007", "depth_m": range(13)})
    grids = {(year, "007"): np.array([n, n + 1.0]) for n, year in enumerate(YEARS)}
    return table, grids


def test_relabels_the_two_depth_inputs_and_keeps_sampling_seed_with_source(inputs):
    table, grids = inputs
    relabeled, distributions, sources = relabel_depth_inputs("permuted_01", table, grids)
    for target_year, source_year in sources.items():
        archived_tract = table.loc[table["year"].eq(source_year), "depth_m"].item()
        assert relabeled.loc[relabeled["year"].eq(target_year), "depth_m"].item() == archived_tract
        np.testing.assert_array_equal(distributions[target_year, "007"], grids[source_year, "007"])
        assert depth_sampling_seed(42, target_year, sources) == 100_042 + source_year
    assert depth_sampling_seed(42, 2011, sources) != depth_sampling_seed(42, 2011, {})
    assert list(table["year"]) == list(YEARS)
    assert depth_sampling_seed(42, 2011, {}) == 102053


def test_explicit_observed_order_matches_unmodified_years_and_seeds(inputs):
    table, grids = inputs
    relabeled, distributions, sources = relabel_depth_inputs("observed", table, grids)
    pd.testing.assert_frame_equal(relabeled, table)
    assert distributions.keys() == grids.keys()
    for year in YEARS:
        np.testing.assert_array_equal(distributions[year, "007"], grids[year, "007"])
        assert depth_sampling_seed(11, year, sources) == depth_sampling_seed(11, year, {})


@pytest.mark.parametrize("truncate_input", ["tract", "grid"])
def test_relabeled_events_require_all_source_years_in_both_inputs(inputs, truncate_input):
    table, grids = inputs
    if truncate_input == "tract":
        table = table.iloc[:-1]
    else:
        del grids[(2023, "007")]
    with pytest.raises(ValueError, match="each 2011–2023 event"):
        relabel_depth_inputs("permuted_01", table, grids)


def test_sequence_cli_requires_an_explicit_output_root():
    with pytest.raises(SystemExit, match="2"):
        run_single_scenario(["--wrr-flood-order", "observed", "--no-plots"])


def test_relabel_rejects_duplicate_tract_year_means(inputs):
    table, grids = inputs
    with pytest.raises(ValueError, match="one mean depth"):
        relabel_depth_inputs("observed", pd.concat([table, table.iloc[[0]]]), grids)


def test_sequence_cli_does_not_allow_observed_year_plots(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(SystemExit, match="2"):
        run_single_scenario(["--wrr-flood-order", "permuted_01", "--out-root", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("flag", ["--compare-flood-or", "--compare-all-models", "--run-all"])
def test_sequence_cli_blocks_batch_paths_before_dispatch(tmp_path, flag):
    output = tmp_path / "output"
    with pytest.raises(SystemExit, match="2"):
        main(["--wrr-flood-order", "permuted_01", "--no-plots", "--out-root", str(output), flag])
    assert not output.exists()


def test_sequence_cli_never_reuses_a_scenario_output(tmp_path):
    output = tmp_path / "baseline" / "baseline"
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="ascii")
    with pytest.raises(FileExistsError):
        run_single_scenario([
            "--wrr-flood-order", "permuted_01", "--no-plots",
            "--scenario", "baseline", "--out-root", str(tmp_path),
        ])
    assert sentinel.read_text(encoding="ascii") == "keep"
    assert list(output.iterdir()) == [sentinel]


def test_sequence_cli_blocks_a_path_valued_yaml_scenario(tmp_path, monkeypatch, capsys):
    out = tmp_path / "output"
    monkeypatch.setattr(runner, "load_yaml_cfg", lambda _: {"scenario": "../outside"})
    with pytest.raises(SystemExit, match="2"):
        run_single_scenario(["--wrr-flood-order", "observed", "--no-plots", "--out-root", str(out)])
    assert "requires a baseline or worst scenario" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_sequence_cli_blocks_an_absolute_registry_model_key(tmp_path, monkeypatch, capsys):
    from utils.model_registry import ModelRegistry

    out = tmp_path / "output"
    outside = tmp_path / "outside"
    monkeypatch.setattr(ModelRegistry, "list_models", lambda _: [str(outside)])
    with pytest.raises(SystemExit, match="2"):
        run_single_scenario([
            "--wrr-flood-order", "observed", "--no-plots", "--scenario", "baseline",
            "--model-dir", str(outside), "--out-root", str(out),
        ])
    assert "requires a model directory inside" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("link_target", ["model", "scenario"])
def test_sequence_cli_cannot_follow_model_or_scenario_links(tmp_path, link_target):
    out = tmp_path / "output"
    model = out / "baseline"
    absent = tmp_path / "outside"
    if link_target == "model":
        link = model
    else:
        link = model / "baseline"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(absent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Creating directory links is not supported here: {error}")
    with pytest.raises((SystemExit if link_target == "model" else FileExistsError)):
        run_single_scenario([
            "--wrr-flood-order", "permuted_01", "--no-plots",
            "--scenario", "baseline", "--out-root", str(out),
        ])
    assert not absent.exists()
