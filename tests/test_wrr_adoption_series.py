import sys

import pandas as pd
import pytest

from scripts.utilities import rebuild_wrr_adoption_series as adoption


def test_adoption_rebuild_counts_held_and_new_adoptions_by_current_tract():
    strata = pd.Series({"001": True, "002": False})
    decisions = pd.DataFrame([
        ("owner", "001", "FI", 0, None),
        ("owner", "001", "DN", 0, " held "),
        ("owner", "001", "EH", 0, None),
        ("owner", "001", "DN", 2, "  "),
        ("owner", "002", "DN", 0, None),
        ("renter", "001", "DN", 0, " held "),
        ("renter", "002", "RL", 0, None),
    ], columns=["group", "tract_geoid", "action", "ELEV_FT", "POLICY_NAME"])
    actual = {(group, action, exposure): value for group, action, exposure, value in
              adoption.adoption_rates(decisions, strata)}
    assert actual == {
        ("homeowner", "FI", "flood-prone"): .5, ("homeowner", "FI", "non-prone"): 0.,
        ("homeowner", "EH", "flood-prone"): .5, ("homeowner", "EH", "non-prone"): 0.,
        ("renter", "FI", "flood-prone"): 1., ("renter", "FI", "non-prone"): 0.,
    }


@pytest.mark.parametrize("renter_row", [
    "1,2011,renter,002,DN,0,",
    "2,2012,renter,002,DN,0,",
    "2,2011,renter,999,DN,0,",
])
def test_read_decisions_rejects_wrong_identifier_year_or_tract(tmp_path, renter_row):
    path = tmp_path / "decisions.csv"
    path.write_text(
        "i,year,group,tract_geoid,action,ELEV_FT,POLICY_NAME\n"
        f"1,2011,owner,001,DN,0,\n{renter_row}\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="Invalid year, decision, or unique household ID"):
        adoption.read_decisions(path, 2011, pd.Index(["001", "002"]))


@pytest.fixture
def draw_rows():
    return [(*key, draw, draw / 100) for key in sorted(adoption.expected_keys(adoption.REBUILT))
            for draw in range(1, adoption.N_RUNS + 1)]


@pytest.fixture
def series(draw_rows):
    rebuilt = adoption.summarize_draws(draw_rows)
    kept = pd.DataFrame(sorted(adoption.expected_keys((adoption.RETAINED,))), columns=adoption.KEYS)
    kept = kept.assign(median=.2, q25=.1, q75=.3,
                       denominator="renters without FI; grouped by initial tract classification")
    return rebuilt, pd.concat([rebuilt, kept], ignore_index=True)


def test_summary_rejects_a_duplicate_draw_cell_or_nan(draw_rows):
    out = adoption.summarize_draws(draw_rows)
    assert len(out) == 78
    assert out.iloc[0][adoption.STATS].tolist() == pytest.approx([.255, .1325, .3775])
    with pytest.raises(ValueError, match="Incomplete or repeated"):
        adoption.summarize_draws([*draw_rows[1:], draw_rows[-1]])
    with pytest.raises(ValueError, match="rate is missing"):
        adoption.summarize_draws([(*draw_rows[0][:-1], float("nan")), *draw_rows[1:]])


def test_assemble_preserves_exact_published_relocation_rows(series):
    rebuilt, reference = series
    out, delta = adoption.assemble_verified(rebuilt, reference)
    assert len(out) == 104 and delta == 0.
    keep = out[out["action"].eq("RL")].reset_index(drop=True)
    expected = reference[reference["action"].eq("RL")][adoption.COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(keep, expected)
    with pytest.raises(ValueError, match="104 published"):
        adoption.assemble_verified(rebuilt, reference.iloc[:-1])


def test_failed_figure8a_c_comparison_leaves_no_new_output(tmp_path, monkeypatch, series):
    rebuilt, reference = series
    path = tmp_path / "reference.csv"
    reference.to_csv(path, index=False)
    rebuilt.loc[0, "median"] = .9
    out = tmp_path / "result"
    monkeypatch.setattr(adoption, "REFERENCE", path)
    monkeypatch.setattr(adoption, "read_strata", lambda _: None)
    monkeypatch.setattr(adoption, "rebuild_series", lambda *_: rebuilt)
    monkeypatch.setattr(sys, "argv", ["prog", "--mc-root", str(tmp_path), "--out-dir", str(out)])
    with pytest.raises(SystemExit, match="2"):
        adoption.main()
    assert not out.exists()


def test_cli_names_which_panels_were_rebuilt_and_which_were_retained(
    tmp_path, monkeypatch, capsys, series
):
    rebuilt, reference = series
    path = tmp_path / "reference.csv"
    reference.to_csv(path, index=False)
    out = tmp_path / "result"
    monkeypatch.setattr(adoption, "REFERENCE", path)
    monkeypatch.setattr(adoption, "read_strata", lambda _: None)
    monkeypatch.setattr(adoption, "rebuild_series", lambda *_: rebuilt)
    monkeypatch.setattr(sys, "argv", ["prog", "--mc-root", str(tmp_path), "--out-dir", str(out)])
    adoption.main()
    log = capsys.readouterr().out
    assert "78 rows rebuilt" in log and "Retained 26 revised Figure 8d rows" in log
    assert list(out.iterdir()) == [out / adoption.FILENAME]
    assert len(pd.read_csv(out / adoption.FILENAME)) == 104
