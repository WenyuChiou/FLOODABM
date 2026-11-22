from collections import Counter
import csv
from io import StringIO

from scripts.utilities import build_wrr_mc_variance_table as assembler


def test_matches_the_published_text_s7_file():
    built = assembler.build()
    assert built == assembler.PUBLISHED.read_text(encoding="utf-8")
    rows = list(csv.DictReader(StringIO(built)))
    assert Counter(row["figure"] for row in rows) == {
        "Figure 5": 39,
        "Figure 6": 260,
        "Figure 8": 104,
        "Figure 9": 52,
    }


def test_distinguishes_paired_error_bars_and_fixed_initial_loss():
    rows = list(csv.DictReader(StringIO(assembler.build())))
    pair = [row for row in rows if row["metric"] == "Premium plus OOP per household"]
    assert len(pair) == 26
    assert all("paired within each run" in row["basis"] for row in pair)
    cumulative = [row for row in rows if row["metric"].startswith("Cumulative ")]
    assert len(cumulative) == 104
    assert {row["scenario"] for row in cumulative} == {"adaptation", "no_adaptation"}
    assert all("initial tenure count" in row["basis"] for row in cumulative)
    assert {row["unit"] for row in rows if row["metric"].endswith("rate")} == {"fraction"}
