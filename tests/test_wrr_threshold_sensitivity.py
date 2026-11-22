"""Verify the fixed-population and same-draw comparisons in Figures S4 and S5."""

import pandas as pd
import pytest

from scripts.utilities import rebuild_wrr_action_intervals as s4
from scripts.utilities import rebuild_wrr_gul_threshold_series as s5


def outcomes(control, case):
    rows = []
    for draw in (1, 2):
        rows.extend([
            (control, draw, "owner", 2011, 10, 100 * draw, 80 * draw),
            (control, draw, "owner", 2012, 10, 200 * draw, 150 * draw),
            (control, draw, "renter", 2011, 5, 50, 40),
            (control, draw, "renter", 2012, 5, 50, 40),
            (case, draw, "owner", 2011, 10, 100 * draw, 80 * draw),
            (case, draw, "owner", 2012, 5, 300 * draw, 200 * draw),
            (case, draw, "renter", 2011, 5, 50, 40),
            (case, draw, "renter", 2012, 5, 750, 450),
        ])
    return pd.DataFrame(rows, columns=[*s4.ANNUAL_KEYS, "households", *s5.SUMMED])


def test_s4_pairs_at_same_draw_and_plots_only_the_action_tenure():
    case = "EX01_EH_higher"
    registry = pd.DataFrame({"case_id": [case], "case_order": [1], "component": ["EH"],
                             "variation": ["higher"], "lower_bound": [.2], "upper_bound": [.3]})
    paired, plot = s4.summarize(outcomes(s4.CONTROL, case), registry, {"owner": 10, "renter": 5})
    assert len(paired) == 8
    assert plot["group"].tolist() == ["owner"]
    assert plot.iloc[0]["GUL_median"] == 15
    assert plot.iloc[0]["actual_loss_median"] == 7.5


def test_s5_uses_initial_household_count_even_after_population_changes():
    case = "EX02_homeowner_020"
    annual = outcomes(s5.CONTROL_ID, case)
    annual["series"] = annual["case_id"].map({s5.CONTROL_ID: "control", case: "homeowner"})
    annual["threshold"] = annual["case_id"].map({s5.CONTROL_ID: .1, case: .2})
    paired, plot = s5.summarize(annual, {"owner": 10, "renter": 5})
    assert paired["GUL_difference"].tolist() == [10, 20]
    assert plot.iloc[0]["GUL_median"] == 15
    assert plot.iloc[0]["GUL_q1"] == 12.5
    assert plot.iloc[0]["GUL_q3"] == 17.5


def test_s5_rejects_a_missing_same_tenure_draw_control():
    case = "EX02_homeowner_020"
    annual = outcomes(s5.CONTROL_ID, case)
    annual["series"] = annual["case_id"].map({s5.CONTROL_ID: "control", case: "homeowner"})
    annual["threshold"] = annual["case_id"].map({s5.CONTROL_ID: .1, case: .2})
    missing = ((annual["case_id"] == s5.CONTROL_ID) & (annual["group"] == "owner")
               & (annual["run_number"] == 2))
    with pytest.raises(ValueError, match="same-draw, same-tenure control"):
        s5.summarize(annual.loc[~missing], {"owner": 10, "renter": 5})


def test_s4_archive_requires_one_unambiguous_finance_folder(tmp_path):
    direct = tmp_path / "finance"
    direct.mkdir()
    assert s4.case_finance(tmp_path) == direct
    (tmp_path / "baseline" / "finance").mkdir(parents=True)
    with pytest.raises(ValueError, match="one recognized archive finance folder"):
        s4.case_finance(tmp_path)
