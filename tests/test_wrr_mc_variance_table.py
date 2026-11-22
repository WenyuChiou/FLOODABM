from collections import Counter
import csv
from decimal import Decimal
from io import StringIO

import pytest

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


def test_rebuilt_table_compares_rows_by_full_identity():
    published = assembler.PUBLISHED.read_text(encoding="utf-8")
    header, *rows = published.splitlines()
    swapped = "\n".join([header, *reversed(rows), ""])
    assert assembler.verify_rebuilt(swapped, published) == 0


def test_rebuilt_table_tolerates_float_serialization_only():
    published = assembler.PUBLISHED.read_text(encoding="utf-8")
    rows = list(csv.DictReader(StringIO(published)))
    start = Decimal(rows[0]["median"])
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=assembler.FIELDS, lineterminator="\n")
    writer.writeheader()
    rows[0]["median"] = str(start + Decimal("1e-14"))
    writer.writerows(rows)
    assert assembler.verify_rebuilt(output.getvalue(), published) == Decimal("1e-14")


def test_rebuilt_table_rejects_changed_metadata():
    published = assembler.PUBLISHED.read_text(encoding="utf-8")
    changed = published.replace("Simulated standardized county payout",
                                "Observed standardized county payout", 1)
    with pytest.raises(ValueError, match="row differs in year, metric, or metadata"):
        assembler.verify_rebuilt(changed, published)


def test_rebuilt_table_rejects_a_larger_quantile_change():
    published = assembler.PUBLISHED.read_text(encoding="utf-8")
    rows = list(csv.DictReader(StringIO(published)))
    rows[0]["median"] = str(Decimal(rows[0]["median"]) + Decimal("1e-8"))
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=assembler.FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with pytest.raises(ValueError, match="annual quantile differs"):
        assembler.verify_rebuilt(output.getvalue(), published)


@pytest.mark.parametrize("extra", [True, False])
def test_rebuilt_table_rejects_ragged_rows(extra):
    published = assembler.PUBLISHED.read_text(encoding="utf-8")
    header, first, *rest = published.splitlines()
    malformed = first + ",unheaded" if extra else first.rsplit(",", 1)[0]
    with pytest.raises(ValueError, match="extra or missing cells"):
        assembler.verify_rebuilt("\n".join([header, malformed, *rest]), published)


def test_rebuilt_cli_requires_both_sources(monkeypatch):
    monkeypatch.setattr("sys.argv", ["assembler", "--check", "--finance-series-dir", "finance"])
    with pytest.raises(SystemExit) as error:
        assembler.main()
    assert error.value.code == 2


@pytest.mark.parametrize("changed_encoding", ["CRLF", "BOM"])
def test_default_check_detects_changed_csv_bytes(monkeypatch, tmp_path, changed_encoding):
    reference = assembler.PUBLISHED.read_bytes()
    altered = reference.replace(b"\n", b"\r\n") if changed_encoding == "CRLF" else b"\xef\xbb\xbf" + reference
    target = tmp_path / "altered.csv"
    target.write_bytes(altered)
    monkeypatch.setattr(assembler, "PUBLISHED", target)
    monkeypatch.setattr("sys.argv", ["assembler", "--check"])
    with pytest.raises(SystemExit) as error:
        assembler.main()
    assert error.value.code == 1
