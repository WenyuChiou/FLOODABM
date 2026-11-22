"""Fixed event orders used in the revised WRR flood-sequence experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd


YEARS = tuple(range(2011, 2024))
# Source events are in target simulation-year order (2011–2023).
# These are the saved experiment orders, not freshly sampled permutations.
EVENT_ORDERS = {
    "observed": YEARS,
    "permuted_01": (2017, 2014, 2018, 2016, 2012, 2015, 2011, 2021, 2019, 2023, 2020, 2022, 2013),
    "permuted_02": (2017, 2016, 2014, 2020, 2015, 2018, 2012, 2013, 2022, 2021, 2023, 2019, 2011),
    "permuted_03": (2019, 2023, 2018, 2013, 2011, 2022, 2020, 2012, 2016, 2021, 2015, 2014, 2017),
    "permuted_04": (2015, 2023, 2017, 2014, 2012, 2021, 2018, 2020, 2011, 2019, 2022, 2016, 2013),
    "permuted_05": (2017, 2018, 2019, 2012, 2020, 2022, 2015, 2021, 2011, 2016, 2014, 2023, 2013),
    "permuted_06": (2022, 2014, 2021, 2013, 2016, 2017, 2012, 2023, 2015, 2019, 2011, 2020, 2018),
    "permuted_07": (2015, 2023, 2022, 2021, 2012, 2016, 2017, 2019, 2014, 2011, 2018, 2020, 2013),
    "permuted_08": (2019, 2022, 2014, 2018, 2017, 2016, 2015, 2023, 2021, 2011, 2013, 2012, 2020),
    "permuted_09": (2016, 2013, 2011, 2015, 2022, 2012, 2019, 2017, 2021, 2020, 2018, 2023, 2014),
    "permuted_10": (2022, 2020, 2016, 2011, 2012, 2021, 2014, 2017, 2018, 2019, 2015, 2023, 2013),
}


def relabel_depth_inputs(
    order: str,
    tract_depths: pd.DataFrame,
    grid_depths: dict[tuple[int, str], np.ndarray],
) -> tuple[pd.DataFrame, dict[tuple[int, str], np.ndarray], dict[int, int]]:
    """Move both depth inputs together; return the source year for each target."""
    source_years = EVENT_ORDERS[order]
    required = set(YEARS)
    if (
        len(source_years) != len(YEARS) or set(source_years) != required
        or set(tract_depths["year"]) != required
        or {year for year, _ in grid_depths} != required
    ):
        raise ValueError("The WRR flood order requires each 2011–2023 event once in both depth inputs")
    if tract_depths.duplicated(["year", "tract_geoid"]).any():
        raise ValueError("The WRR flood order requires one mean depth per tract and year")
    source_by_target = dict(zip(YEARS, source_years, strict=True))
    target_by_source = {source: target for target, source in source_by_target.items()}
    relabeled_tracts = tract_depths.assign(year=tract_depths["year"].map(target_by_source).astype(int))
    relabeled_grids = {
        (target_by_source[year], tract): values
        for (year, tract), values in grid_depths.items()
    }
    return relabeled_tracts, relabeled_grids, source_by_target


def depth_sampling_seed(seed: int, target_year: int, source_by_target: dict[int, int]) -> int:
    """Keep the sampling stream attached to its flood event, not its time slot."""
    return seed + 100_000 + source_by_target.get(target_year, target_year)
