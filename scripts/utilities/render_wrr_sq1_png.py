"""Rebuild the accepted six-panel SQ1 PNG from aggregate simulation series."""

from io import BytesIO
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

if __package__:
    from .rebuild_wrr_finance_series import ANNUAL_NAME, CUMULATIVE_NAME, TABLE_SPECS, compare_series
    from .render_wrr_vector_figures import render_sq1, verify_pixels
    from .verify_wrr_figures import REFERENCE, ROOT
else:
    from rebuild_wrr_finance_series import ANNUAL_NAME, CUMULATIVE_NAME, TABLE_SPECS, compare_series
    from render_wrr_vector_figures import render_sq1, verify_pixels
    from verify_wrr_figures import REFERENCE, ROOT


YEARS = tuple(range(2011, 2024))
X = np.arange(len(YEARS))
RED, BLUE = "#d62828", "#0077b6"
STYLE = {
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 12,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 13,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 10.5,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.12,
    "grid.linestyle": "--", "axes.spines.top": False, "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 3,
}
FINANCE_STYLE = {
    **STYLE, "axes.titlesize": 16, "axes.labelsize": 14, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 11, "axes.spines.right": False,
}
# Fixed point coordinates from the accepted 3×2 layout. The AEP panels were
# moved to Figure 7; keeping these extents avoids redrawing them just for layout.
FINANCE_SIZE = (851.938125, 889.028716)
FINANCE_LEFT = (57.934375, 482.089375)
FINANCE_BOTTOM = (633.2075, 337.51625)
FINANCE_PANEL = (300.305, 223.8695588235)


def dollars(value, _):
    if abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.0f}K"
    return f"${value:.0f}"


def annual_ticks(ax):
    ax.set_xticks(X[::2])
    ax.set_xticklabels([str(y) for y in YEARS[::2]], rotation=0)
    ax.set_xlim(X[0] - 0.7, X[-1] + 1.2)


def major_floods(ax):
    for year in (2011, 2021):
        index = YEARS.index(year)
        ax.axvspan(index - 0.45, index + 0.45, color="#e0e0e0", alpha=0.55, zorder=0)


def panel_letter(ax, text, size):
    ax.text(-0.10, 1.04, text, transform=ax.transAxes, fontsize=size,
            fontweight="bold", ha="left", va="bottom")


def value_band(rows):
    """Read one complete annual series as median, lower and upper quartiles."""
    rows = rows.sort_values("year")
    if tuple(rows.year) != YEARS:
        raise ValueError("Expected one value for every year 2011–2023 in each Figure 6 series")
    values = rows[["median", "q25", "q75"]].to_numpy(dtype=float).T
    if not np.isfinite(values).all():
        raise ValueError("Figure 6 plotting values must be finite")
    return values


def whisker_lengths(center, lo, hi):
    error = np.vstack([center - lo, hi - center])
    # When the stacked bar and its IQR coincide, adding the two medians can
    # round differently from summarizing their total. Remove only this
    # floating-point noise in the plotted lengths, not in the underlying data.
    tolerance = 4 * np.spacing(np.maximum(np.abs(center), np.abs(np.vstack([lo, hi]))))
    if np.any(error < -tolerance):
        raise ValueError("Figure 6 bar or line center is outside the IQR")
    return np.maximum(error, 0)


def whiskers(ax, median, lo, hi, *, caps, thickness, zorder):
    ax.errorbar(X, median, yerr=whisker_lengths(median, lo, hi),
                fmt="none", ecolor="black", elinewidth=1.2,
                capsize=caps, capthick=thickness, zorder=zorder)


def money_ticks(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(dollars))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))


def cumulative_figure(data):
    fig = plt.figure(figsize=(10.5, 4.8))
    grid = fig.add_gridspec(1, 2, wspace=0.28, width_ratios=[1, 1])
    for col, (group, title) in enumerate((("owner", "Homeowner"), ("renter", "Renter"))):
        ax = fig.add_subplot(grid[0, col])
        major_floods(ax)
        subset = data.loc[data.group.eq(group)]
        lines, medians = {}, {}
        for scenario in ("adaptation", "no_adaptation"):
            active = scenario == "adaptation"
            for metric, marker, color, metric_label in (
                ("gul", "o", RED, "GUL"), ("actual", "s", BLUE, "Actual loss")
            ):
                median, lo, hi = value_band(subset.loc[subset.scenario.eq(scenario) & subset.metric.eq(metric)])
                medians[scenario, metric] = median
                if active:
                    whiskers(ax, median, lo, hi, caps=5, thickness=1.5, zorder=8)
                line, = ax.plot(X, median, marker=marker, ms=5, color=color,
                                lw=2.2 if active else 1.8, ls="-" if active else "--",
                                markerfacecolor=color if active else "white",
                                markeredgewidth=1.0 if active else 1.5,
                                label=f"{'Adaptation' if active else 'No-adaptation'} — {metric_label}",
                                zorder=4 if active else 3)
                lines[scenario, metric] = line

        for metric, color in (("gul", RED), ("actual", BLUE)):
            ax.fill_between(X, medians["adaptation", metric], medians["no_adaptation", metric],
                            alpha=0.08, color=color, zorder=1)
        # The final wording is placed after layout so the panel extents are retained.
        ax.set_ylabel("Cumulative loss per initial HH ($)", fontsize=13)
        money_ticks(ax)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        annual_ticks(ax)
        ax.set_xlabel("Year", fontsize=13)
        ax.spines["right"].set_visible(False)
        panel_letter(ax, f"({chr(97 + col)})", 13)
        ax.legend(handles=[lines[s, m] for m in ("gul", "actual") for s in ("adaptation", "no_adaptation")],
                  loc="upper left", fontsize=10.5, framealpha=0.95, edgecolor="#cccccc",
                  handlelength=2.5, labelspacing=0.3, borderpad=0.4, handletextpad=0.4)
        endpoints = sorted(
            (medians[s, m][-1], color) for m, color in (("gul", RED), ("actual", BLUE))
            for s in ("no_adaptation", "adaptation")
        )
        y = [value for value, _ in endpoints]
        min_gap = np.ptp(ax.get_ylim()) * 0.05
        for i in range(1, len(y)):
            y[i] = max(y[i], y[i - 1] + min_gap)
        for i in range(len(y) - 2, -1, -1):
            y[i] = min(y[i], y[i + 1] - min_gap)
        for (value, color), pos in zip(endpoints, y, strict=True):
            ax.text(X[-1] + 0.3, pos, f"${value / 1e3:.0f}K",
                    fontsize=11, fontweight="bold", color=color,
                    ha="left", va="center", clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.85))
    return fig


def finance_figure(data):
    width, height = FINANCE_SIZE
    panel_width, panel_height = FINANCE_PANEL
    fig = plt.figure(figsize=(width / 72, height / 72), layout="none")
    groups = ("homeowner", "renter")
    rates = {}
    by_group = {}
    for group in groups:
        by_group[group] = data.loc[data.group.eq(group)]
        for metric in ("OOP rate", "Payout rate"):
            rates[group, metric] = value_band(by_group[group].loc[by_group[group].metric.eq(metric)]) * 100
    oop_limit = max(20, 1.35 * max(rates[group, "OOP rate"][0].max() for group in groups))
    colors = (("#1a6faf", "#9ecae1", "#08519c"), ("#1f7a3c", "#a1d99b", "#006d2c"))
    primary = {}
    for row, bottom in enumerate(FINANCE_BOTTOM):
        for col, left in enumerate(FINANCE_LEFT):
            primary[row, col] = fig.add_axes((left / width, bottom / height,
                                             panel_width / width, panel_height / height))
    for row in range(2):
        for col, group in enumerate(groups):
            ax = primary[row, col]
            series = by_group[group]

            def band(metric):
                return value_band(series.loc[series.metric.eq(metric)])

            c_prem, c_oop, c_pay = colors[col]
            major_floods(ax)
            if row == 0:
                prem, _, _ = band("Premium per household")
                oop, _, _ = band("OOP cost per household")
                top = prem + oop
                _, lo, hi = band("Premium plus OOP per household")
                ax.bar(X, prem, 0.65, label="Premium", color=c_prem, alpha=0.85, zorder=3)
                ax.bar(X, oop, 0.65, bottom=prem, label="OOP cost", color=c_oop, alpha=0.85, zorder=3)
                ax.set_ylabel("Average per HH ($)", fontsize=14)
                rate_name, marker, rate_limit, ticks = "OOP rate", "o", oop_limit, 4
            else:
                top, lo, hi = band("Payout per household")
                ax.bar(X, top, 0.65, label="Payout", color=c_pay, alpha=0.85, zorder=3)
                ax.set_ylabel("Average payout per HH ($)", fontsize=14)
                ax.set_xlabel("Year", fontsize=14)
                rate_name, marker, rate_limit, ticks = "Payout rate", "s", 102, 5
            whiskers(ax, top, lo, hi, caps=4, thickness=1.2, zorder=6)
            money_ticks(ax)
            ax.set_title(group.capitalize(), fontsize=16, fontweight="bold", pad=8)
            annual_ticks(ax)
            ax.set_xlim(X[0] - 0.6, X[-1] + 0.6)
            panel_letter(ax, f"({chr(97 + 2 * row + col)})", 18)
            twin = ax.twinx()
            rate, low_rate, high_rate = rates[group, rate_name]
            twin.fill_between(X, low_rate, high_rate, color="#777777", alpha=0.25, zorder=4, linewidth=0)
            twin.plot(X, rate, color="#444444", marker=marker, ms=4, lw=1.6,
                      linestyle="--", label=rate_name, zorder=5)
            twin.set_ylabel("Share of GUL (%)", fontsize=18, color="#444444")
            twin.tick_params(axis="y", labelsize=15, colors="#444444")
            twin.spines["top"].set_visible(False)
            twin.yaxis.set_major_locator(MaxNLocator(nbins=ticks, prune="both"))
            twin.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
            twin.set_ylim(0, rate_limit)
            if row == 0 and col == 1:
                ax.set_ylim(top=float((top + (hi - top)).max()) * 1.35)
            handles, labels = ax.get_legend_handles_labels()
            rate_handles, rate_labels = twin.get_legend_handles_labels()
            ax.legend(handles + rate_handles, labels + rate_labels,
                      loc="upper right" if row == 0 and col == 1 else "upper left",
                      fontsize=11, framealpha=0.92, edgecolor="#cccccc", handlelength=1.8,
                      labelspacing=0.3, borderpad=0.4)
    return fig


def in_memory_vector(pdf, fig, **save_options):
    with BytesIO() as surface:
        try:
            fig.savefig(surface, format="pdf", **save_options)
            return pdf.open(stream=surface.getvalue(), filetype="pdf")
        finally:
            plt.close(fig)


def retitle(pdf, page, fonts, replacements, legend_terms=()):
    """Apply the accepted label edits without moving axes or plotting marks."""
    selected = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                name = span["text"]
                if name in replacements or name in legend_terms:
                    selected.append((line, span, replacements.get(name, name)))
                    page.add_redact_annot(pdf.Rect(span["bbox"]), fill=False)
    for name in replacements:
        if not any(span["text"] == name for _, span, _ in selected):
            raise ValueError(f"Expected Figure 6 label: {name}")
    page.apply_redactions(images=0, graphics=0, text=0)
    for alias, content in zip(("WRRRegular", "WRRBold"), fonts, strict=True):
        page.insert_font(fontname=alias, fontbuffer=content)
    for line, span, text in selected:
        vertical = abs(line["dir"][1]) > 0.5
        size = 12 if span["text"] in legend_terms else span["size"]
        page.insert_text(span["origin"], text, fontsize=size, rotate=90 if vertical else 0,
                         fontname="WRRBold" if "Bold" in span["font"] else "WRRRegular")


def read_series(name: str, series_dir: Path | None) -> pd.DataFrame:
    bundled = ROOT / "data" / "figures"
    opts = {"encoding": "utf-8-sig", "float_precision": "round_trip"}
    expected = pd.read_csv(bundled / name, **opts)
    if series_dir is None or series_dir == bundled:
        return expected
    rebuilt = pd.read_csv(series_dir / name, **opts)
    compare_series(rebuilt, expected, *TABLE_SPECS[name])
    return rebuilt


def verified_png(series_dir: Path | None = None, font_dir: Path | None = None) -> bytes:
    """Verify the final six-panel pixels before the CLI writes any files."""
    import pymupdf as pdf

    if pdf.VersionBind != "1.26.7" or mpl.__version__ != "3.10.7":
        raise ValueError("Use Matplotlib 3.10.7 and PyMuPDF 1.26.7 from the WRR requirements files")
    if font_dir is None:
        windows = os.environ.get("WINDIR")
        if not windows:
            raise ValueError("Supply --font-dir with the local regular and bold Times New Roman fonts")
        font_dir = Path(windows) / "Fonts"
    fonts = tuple((font_dir / name).read_bytes() for name in ("times.ttf", "timesbd.ttf"))
    cumulative = read_series(CUMULATIVE_NAME, series_dir)
    finance = read_series(ANNUAL_NAME, series_dir)
    with mpl.rc_context(STYLE), in_memory_vector(pdf, cumulative_figure(cumulative), bbox_inches="tight") as top:
        retitle(pdf, top[0], fonts, {"Cumulative loss per initial HH ($)": "Cumulative loss per household ($)"})
        with mpl.rc_context(FINANCE_STYLE), in_memory_vector(pdf, finance_figure(finance)) as bottom:
            retitle(pdf, bottom[0], fonts, {
                "Average per HH ($)": "Average per household ($)",
                "Average payout per HH ($)": "Average payout per household ($)",
            }, {"Premium", "OOP cost", "OOP rate", "Payout", "Payout rate"})
            png = render_sq1(pdf, top, bottom, fonts[1])
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))["figures"]["Figure_06.png"]
    verify_pixels(png, expected)
    return png
