from copy import deepcopy
import csv
from io import StringIO

import pandas as pd
import pytest

from main import main, run_single_scenario
from modules.wrr_action_intervals import CASES, CONTROL_BOUNDS, apply_wrr_action_interval
from scripts.utilities import rebuild_wrr_action_intervals as s4
from scripts.utilities import rebuild_wrr_gul_threshold_series as s5
from scripts.utilities import run_wrr_threshold_sensitivity as replay
from scripts.utilities import wrr_threshold_cases as design


def test_all_s4_cases_match_revised_plot_and_only_change_their_own_interval():
    rows = list(csv.DictReader(StringIO(design.bundled_plot(4).decode("utf-8"))))
    execution_order = sorted(rows, key=lambda row: int(row["case_order"]))
    assert [row["case_id"] for row in execution_order] == list(CASES) == list(design.CASE_IDS[4])
    base = {"draw_bounds": {key: list(value) for key, value in CONTROL_BOUNDS.items()},
            "sfha_initialization": {"enabled": True, "fi_draw_bounds_inside": [0., .1]}}
    for row in rows:
        case_id = row["case_id"]
        key = "FI" if row["component"] == "FI_homeowner_outside_SFHA" else row["component"]
        config = deepcopy(base)
        expected = deepcopy(base)
        expected["draw_bounds"][key] = [float(row["lower_bound"]), float(row["upper_bound"])]
        apply_wrr_action_interval(config, case_id)
        assert config == expected


def test_s4_rejects_noncanonical_control_or_sfha():
    case = "EX01_EH_lower"
    config = {"draw_bounds": dict(CONTROL_BOUNDS),
              "sfha_initialization": {"enabled": True, "fi_draw_bounds_inside": [0., .2]}}
    with pytest.raises(ValueError, match="inside-SFHA"):
        apply_wrr_action_interval(config, case)
    config["sfha_initialization"]["fi_draw_bounds_inside"] = [0., .1]
    config["draw_bounds"]["RL"] = [.2, .8]
    with pytest.raises(ValueError, match="control action intervals"):
        apply_wrr_action_interval(config, case)


@pytest.mark.parametrize("flags", [
    ["--wrr-flood-order", "observed"], ["--thr-renter", ".20"],
    ["--scenario", "worst"], ["--deterministic"], ["--years", "12"],
    ["--compare-all-models"], ["--run-all"],
])
def test_s4_cli_rejects_scientific_or_batch_overrides_before_writing(tmp_path, flags):
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="2"):
        main(["--wrr-s4-case", "EX01_EH_lower", "--no-plots", "--out-root", str(out), *flags])
    assert not out.exists()


def test_s4_cli_preserves_an_existing_output(tmp_path):
    out = tmp_path / "baseline" / "baseline"
    out.mkdir(parents=True)
    marker = out / "existing.txt"
    marker.write_text("intact", encoding="ascii")
    with pytest.raises(FileExistsError):
        run_single_scenario(["--wrr-s4-case", "EX01_EH_lower", "--no-plots", "--out-root", str(tmp_path)])
    assert list(out.iterdir()) == [marker]
    assert marker.read_text(encoding="ascii") == "intact"


def test_bundled_threshold_references_and_both_tenures():
    assert len(design.load_hashes(4)) == 200
    assert len(design.load_hashes(5)) == 160
    assert design.case_model_args(5, "EX02_homeowner_020") == ("--thr-owner", "0.2", "--thr-renter", "0.1")
    assert design.case_model_args(5, "EX02_renter_090") == ("--thr-owner", "0.1", "--thr-renter", "0.9")
    with pytest.raises(ValueError, match="Not a revised"):
        design.case_output_root(5, "../../outside", 1)


@pytest.mark.parametrize("number", [4, 5])
def test_aggregators_parse_verified_control_bytes_even_if_path_is_replaced(tmp_path, monkeypatch, number):
    root = tmp_path.resolve()
    finance = root / "run_01" / "baseline" / "baseline" / "finance"
    finance.mkdir(parents=True)
    accepted = {year: f"control\n{year}\n".encode() for year in design.YEARS}
    for year, payload in accepted.items():
        (finance / f"finance_tract_{year}.csv").write_bytes(payload)
    monkeypatch.setattr(design, "DRAW_IDS", (1,))
    monkeypatch.setattr(design, "reference_hashes",
                        lambda _: {(1, "baseline"): design.case_digest(accepted)})
    controls = design.snapshot_controls(root, root / "unused-reference.csv")
    (finance / "finance_tract_2011.csv").write_bytes(b"replaced after validation")
    with pytest.raises(ValueError, match="baseline control differs"):
        design.snapshot_controls(root, root / "unused-reference.csv")

    producer = s4 if number == 4 else s5
    monkeypatch.setattr(producer, "DRAW_IDS", (1,))

    def fake_parse(path, *, payload=None):
        if path.parent == finance:
            assert payload == accepted[int(path.stem[-4:])]
        return {"owner": {"households": 1, "gul": 2, "payout": 1}}, None

    monkeypatch.setattr(producer, "read_tract_finance", fake_parse)
    if number == 4:
        monkeypatch.setattr(s4, "case_finance", lambda target: target / "finance")
        metadata = pd.DataFrame({"case_id": design.CASE_IDS[4]})
        annual = s4.rebuild_annual(root / "cases", root, metadata, controls)
        control_id = s4.CONTROL
    else:
        annual = s5.rebuild_annual(root / "cases", root, controls)
        control_id = s5.CONTROL_ID
    assert annual.loc[annual["case_id"].eq(control_id), "actual_loss_total_usd"].tolist() == [1] * 13


@pytest.mark.parametrize(("number", "case"), [(4, "EX01_EH_lower"), (5, "EX02_renter_090")])
def test_threshold_runner_rechecks_output_before_reuse(tmp_path, monkeypatch, number, case):
    root = tmp_path.resolve()
    draw = 9
    data = {year: f"data\n{year}\n".encode() for year in design.YEARS}
    sha = design.case_digest(data)
    target = root / design.case_output_root(number, case, draw)
    finance = target / "baseline" / "baseline" / "finance"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == design.ROOT and kwargs["check"] and kwargs["timeout"] == 90
        assert kwargs["env"]["FLOODABM_POSTERIOR_IDX"] == "768"
        assert command[-len(design.case_model_args(number, case)):] == list(design.case_model_args(number, case))
        finance.mkdir(parents=True)
        for year, content in data.items():
            (finance / f"finance_tract_{year}.csv").write_bytes(content)

    monkeypatch.setattr(replay.subprocess, "run", fake_run)
    assert replay.replay_case(root, number, case, draw, sha, 90, False) == "REPLAYED AND VERIFIED"
    assert replay.replay_case(root, number, case, draw, sha, 90, True) == "VERIFIED EXISTING"
    (finance / "finance_tract_2011.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="finance differs"):
        replay.replay_case(root, number, case, draw, sha, 90, True)
    assert len(calls) == 1
