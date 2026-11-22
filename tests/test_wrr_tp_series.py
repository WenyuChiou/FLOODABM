import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.utilities import rebuild_wrr_tp_series as tp


def test_weights_use_each_current_tenure_population_and_skip_vacant_tracts():
    strata = pd.Series({"001": True, "002": True, "003": False, "004": False})
    counts = pd.Series(
        [3, 1, 1, 1, 1, 2],
        index=pd.MultiIndex.from_tuples([
            ("owner", "001"), ("owner", "002"), ("owner", "003"),
            ("renter", "001"), ("renter", "002"), ("renter", "003"),
        ]),
    )
    post_action = pd.DataFrame(
        {"TP_owner": [.2, .6, .3, np.nan], "TP_renter": [.1, .7, .4, np.nan]},
        index=strata.index,
    )
    values = {(group, exposure): value for group, exposure, value in
              tp.weighted_tp(counts, post_action, strata)}
    assert values == pytest.approx({
        ("homeowner", "flood-prone"): .3, ("homeowner", "non-prone"): .3,
        ("renter", "flood-prone"): .4, ("renter", "non-prone"): .4,
    })
    post_action.loc["002", "TP_renter"] = np.nan
    values = {(group, exposure): value for group, exposure, value in
              tp.weighted_tp(counts, post_action, strata)}
    assert np.isnan(values["renter", "flood-prone"])
    assert values["homeowner", "flood-prone"] == pytest.approx(.3)


@pytest.mark.parametrize(("owner_id", "owner_year"), [("1", 2011), ("2", 2012)])
def test_decision_counts_reject_duplicate_households_and_wrong_year(tmp_path, owner_id, owner_year):
    path = tmp_path / "decisions.csv"
    path.write_text(
        f"i,year,group,tract_geoid\n{owner_id},{owner_year},owner,001\n1,2011,renter,001\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="Invalid tenure, tract, year or unique household ID"):
        tp.read_counts(path, 2011, pd.Index(["001"]))


def test_post_action_source_rejects_pre_action_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "YEARS", range(2011, 2012))
    path = tmp_path / "tp_traj.csv"
    path.write_text("year,tract_geoid,phase,TP_owner,TP_renter\n2011,001,before,.2,.4\n", encoding="ascii")
    with pytest.raises(ValueError, match="post-action TP row"):
        tp.read_post_action_tp(path, pd.Index(["001"]))


@pytest.fixture
def draws():
    return [
        (year, group, exposure, draw, np.nan if key == tp.PUBLISHED_MISSING else draw / 100)
        for year in tp.YEARS
        for group in tp.TENURE.values()
        for exposure in tp.EXPOSURE.values()
        for draw in range(1, tp.N_RUNS + 1)
        for key in [(year, group, exposure, draw)]
    ]


def test_quantiles_keep_only_the_revised_missing_draw(draws):
    result = tp.summarize_draws(draws)
    assert len(result) == 52
    year, group, exposure, _ = tp.PUBLISHED_MISSING
    target = result.set_index(tp.KEYS).loc[year, group, exposure]
    assert target.to_dict() == pytest.approx({"median": .25, "q25": .13, "q75": .38})
    assert result.iloc[0][tp.STATS].tolist() == pytest.approx([.255, .1325, .3775])
    with pytest.raises(ValueError, match="Incomplete or repeated"):
        tp.summarize_draws(draws[1:] + [draws[-1]])
    with pytest.raises(ValueError, match="Missing TP draws"):
        tp.summarize_draws([(*draws[0][:-1], np.nan), *draws[1:]])


def test_publication_comparison_checks_values_and_keys(draws):
    expected = tp.summarize_draws(draws)
    actual = expected.copy()
    assert tp.verify_series(actual, expected.iloc[::-1]) == 0
    actual.loc[0, "q75"] += .01
    with pytest.raises(ValueError, match="statistic differs"):
        tp.verify_series(actual, expected)
    with pytest.raises(ValueError, match="keys differ"):
        tp.verify_series(actual.iloc[1:], expected)


def test_cli_rejects_reference_mismatch_without_creating_output(tmp_path, monkeypatch, draws):
    expected = tp.summarize_draws(draws)
    reference = tmp_path / "reference.csv"
    expected.to_csv(reference, index=False)
    actual = expected.copy()
    actual.loc[0, "median"] = .9
    out = tmp_path / "result"
    monkeypatch.setattr(tp, "REFERENCE", reference)
    monkeypatch.setattr(tp, "read_strata", lambda _: None)
    monkeypatch.setattr(tp, "rebuild_series", lambda *_: actual)
    monkeypatch.setattr(sys, "argv", ["prog", "--mc-root", str(tmp_path), "--out-dir", str(out)])
    with pytest.raises(SystemExit, match="2"):
        tp.main()
    assert not out.exists()


def test_cli_does_not_read_or_modify_an_existing_output(tmp_path, monkeypatch):
    sentinel = tmp_path / tp.FILENAME
    sentinel.write_text("keep", encoding="ascii")
    monkeypatch.setattr(sys, "argv", ["prog", "--mc-root", str(tmp_path), "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="2"):
        tp.main()
    assert sentinel.read_text(encoding="ascii") == "keep"


def test_short_write_does_not_publish_a_partial_series(tmp_path, monkeypatch):
    write_bytes = Path.write_bytes

    def write_short(path, payload):
        return write_bytes(path, payload[:3])

    out = tmp_path / "result"
    monkeypatch.setattr(Path, "write_bytes", write_short)
    with pytest.raises(OSError, match="Incomplete write"):
        tp.publish(out, b"year,group")
    assert not list(tmp_path.iterdir())


def test_publication_does_not_replace_a_concurrent_target(tmp_path, monkeypatch):
    link = tp.os.link

    def race(staged, target):
        target.write_bytes(b"someone else's complete file")
        link(staged, target)

    out = tmp_path / "result"
    monkeypatch.setattr(tp.os, "link", race)
    with pytest.raises(FileExistsError):
        tp.publish(out, b"year,group")
    assert (out / tp.FILENAME).read_bytes() == b"someone else's complete file"
    assert list(tmp_path.iterdir()) == [out]
