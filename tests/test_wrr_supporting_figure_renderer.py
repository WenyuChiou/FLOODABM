import hashlib

import pandas as pd
import pytest

from scripts.utilities import render_wrr_supporting_figures as renderer


@pytest.mark.parametrize("number,rows", [(4, 20), (5, 16), (6, 104)])
def test_bundled_plot_series_matches_pinned_reference(number, rows):
    assert len(renderer.check_series(number, renderer.accepted_csv(number))) == rows


def test_s4_series_from_builder_uses_the_accepted_drawing_order(tmp_path):
    path = tmp_path / "Figure_R1_data.csv"
    expected = pd.read_csv(renderer.accepted_csv(4))
    rebuilt = expected.sort_values("case_order")
    assert rebuilt["case_id"].tolist() != expected["case_id"].tolist()
    rebuilt.to_csv(path, index=False)
    data = renderer.check_series(4, path)
    assert data["case_id"].tolist() == expected["case_id"].tolist()


def test_rebuilt_series_with_duplicate_rows_is_rejected(tmp_path):
    path = tmp_path / "duplicate_case.csv"
    data = pd.read_csv(renderer.accepted_csv(5))
    data.iloc[1] = data.iloc[0]
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="row identities"):
        renderer.check_series(5, path)


def test_rebuilt_series_with_changed_losses_is_rejected(tmp_path):
    path = tmp_path / "wrong_loss.csv"
    data = pd.read_csv(renderer.accepted_csv(6))
    data.loc[0, "center"] += 0.25
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="values differ"):
        renderer.check_series(6, path)


def test_missing_explicit_series_does_not_fall_back(tmp_path):
    out_dir = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        renderer.render_checked(out_dir, {4: tmp_path / "missing.csv"})
    assert not out_dir.exists()


def test_modified_bundled_series_is_rejected_before_render(tmp_path, monkeypatch):
    path = tmp_path / "corrupted_reference.csv"
    path.write_bytes(renderer.accepted_csv(4).read_bytes() + b"\n")
    monkeypatch.setattr(renderer, "accepted_csv", lambda _: path)
    with pytest.raises(ValueError, match="reference CSV bytes differ"):
        renderer.check_series(4, path)


def test_existing_png_is_not_overwritten(tmp_path):
    target = tmp_path / "FigS04_action_threshold_intervals.png"
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        renderer.render_checked(tmp_path, {4: renderer.accepted_csv(4)})
    assert target.read_bytes() == b"keep"


def test_no_pngs_are_written_if_any_render_fails_verification(tmp_path, monkeypatch):
    fixture = b"synthetic-rendered-image"
    monkeypatch.setattr(renderer, "render", lambda *_: fixture)
    monkeypatch.setitem(
        renderer.REFERENCES,
        4,
        (renderer.REFERENCES[4][0], hashlib.sha256(fixture).hexdigest()),
    )
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="S5 PNG differs"):
        renderer.render_checked(out_dir, {n: renderer.accepted_csv(n) for n in (4, 5)})
    assert not out_dir.exists()
