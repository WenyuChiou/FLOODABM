from __future__ import annotations

import numpy as np
import pandas as pd

from modules.actions.decision import sequential_decision_fast


def _state() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["owner", "owner", "renter", "renter"],
            "inside_SFHA": [1, 0, 1, 0],
            "eh_done": [False, False, False, False],
        }
    )


def _probabilities(fi_probability: float) -> dict[str, np.ndarray]:
    return {
        "FI": np.full(4, fi_probability),
        "EH": np.zeros(4),
        "BP": np.zeros(4),
        "RL": np.zeros(4),
    }


def test_low_sfha_fi_interval_applies_only_to_homeowners() -> None:
    result = sequential_decision_fast(
        _state(),
        _probabilities(0.20),
        np.random.RandomState(42),
        draw_bounds={
            "FI": (0.35, 0.55),
            "FI_renter": (0.70, 0.90),
            "FI_inside_SFHA": (0.00, 0.10),
            "EH": (0.30, 0.60),
            "BP": (0.25, 0.65),
            "RL": (0.30, 0.95),
        },
    )

    assert result["action"].tolist() == ["FI", "DN", "DN", "DN"]


def test_renter_fi_interval_does_not_change_homeowner_bounds() -> None:
    result = sequential_decision_fast(
        _state(),
        _probabilities(0.60),
        np.random.RandomState(42),
        draw_bounds={
            "FI": (0.35, 0.55),
            "FI_renter": (0.70, 0.90),
            "FI_inside_SFHA": (0.00, 0.10),
            "EH": (0.30, 0.60),
            "BP": (0.25, 0.65),
            "RL": (0.30, 0.95),
        },
    )

    assert result["action"].tolist() == ["FI", "FI", "DN", "DN"]
