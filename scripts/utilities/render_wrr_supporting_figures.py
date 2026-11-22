"""Redraw and verify the three revised WRR sensitivity figures (S4–S6)."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "figures"
# The September 15, 2026 high-resolution outputs (final_submission).
# These SHA-256 values include PNG encoding, not just the decoded pixels.
REFERENCES = {
    4: (
        "action_threshold_intervals",
        "6e1a282a1b2c3387ae4aa47138ba76d7561a9ad4dfda67aafc68594d6520e9a8",
    ),
    5: (
        "gul_ratio_threshold",
        "bf33aaa9c06922eb08b5e443a9b69521bc9d62627df772d03f6682fbc2fdb697",
    ),
    6: (
        "flood_event_sequence",
        "cbabfd8dd822422a0edea949271ac3476ddff875511d9f42cd517753b4b5b640",
    ),
}
# Canonical LF copies of the three accepted group-level plotting tables.
CSV_SHA256 = {
    4: "c1f14f4ff886ebab613be57690fd92cfb37d1fe1c3f58882d684cbc49acfaf64",
    5: "39b2a0a8a1a5f8ced2cbcd6b3dc25caea1d975e86e209c0e394f34c36c612cf9",
    6: "0124047f38d3733c098d953960f6b7280d096dd5fae3c42f44d245ffd960ccf5",
}
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 17,
    # The archive plots ran in succession. S5/S6 inherit these two settings from S4.
    "xtick.labelsize": 14,
    "ytick.labelsize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
ACTION_COLORS = {
    "FI_homeowner_outside_SFHA": "#0072B2",
    "EH": "#E69F00",
    "BP": "#CC79A7",
    "FI_renter": "#0072B2",
    "RL": "#009E73",
}
COLORS = {"GUL": "#d62728", "actual_loss": "#0072B2"}
COMPONENTS = ["FI_homeowner_outside_SFHA", "EH", "BP", "FI_renter", "RL"]
VARIATIONS = ["lower", "higher", "narrower", "wider"]
ACTION_LABEL = {
    "FI_homeowner_outside_SFHA": "FI outside SFHA",
    "EH": "EH",
    "BP": "BP",
    "FI_renter": "FI",
    "RL": "RL",
}
CURRENT = {
    "FI_homeowner_outside_SFHA": (0.35, 0.55),
    "EH": (0.30, 0.60),
    "BP": (0.25, 0.65),
    "FI_renter": (0.70, 0.90),
    "RL": (0.30, 0.95),
}


def interval_label(lower: float, upper: float) -> str:
    return f"U({lower:.2f}, {upper:.2f})"


def layout_positions() -> tuple[dict[str, float], dict[str, float], float]:
    positions: dict[str, float] = {}
    centers: dict[str, float] = {}
    cursor = 0.0
    for index, component in enumerate(COMPONENTS):
        component_positions: list[float] = []
        for variation in VARIATIONS:
            case_id = f"EX01_{component}_{variation}"
            positions[case_id] = cursor
            component_positions.append(cursor)
            cursor += 1.0
        centers[component] = float(np.mean(component_positions))
        cursor += 2.2 if index == 2 else 1.0
    return positions, centers, cursor - 1.0


def action_intervals(data: pd.DataFrame) -> Figure:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 17.5,
            "xtick.labelsize": 14,
            "ytick.labelsize": 15,
        }
    )
    positions, centers, last_position = layout_positions()
    indexed = data.set_index("case_id")
    figure, axes = plt.subplots(2, 1, figsize=(16.5, 10.0), sharex=True)
    panels = [
        (axes[0], "GUL", "(a) Cumulative GUL difference per household"),
        (axes[1], "actual_loss", "(b) Cumulative actual loss difference per household"),
    ]

    divider = (positions["EX01_BP_wider"] + positions["EX01_FI_renter_lower"]) / 2
    for axis, metric, title in panels:
        medians = indexed[f"{metric}_median"].to_numpy(dtype=float) / 1000
        q1 = indexed[f"{metric}_q1"].to_numpy(dtype=float) / 1000
        q3 = indexed[f"{metric}_q3"].to_numpy(dtype=float) / 1000
        full_min = float(np.min(q1))
        full_max = float(np.max(q3))
        span = max(full_max - full_min, 1.0)
        label_offset = max(0.12, span * 0.018)

        for case_id, median, low, high in zip(indexed.index, medians, q1, q3):
            row = indexed.loc[case_id]
            x = positions[case_id]
            component = str(row["component"])
            yerr = np.array([[median - low], [high - median]])
            axis.bar(
                x,
                median,
                yerr=yerr,
                width=0.78,
                color=ACTION_COLORS[component],
                edgecolor="black",
                linewidth=0.55,
                alpha=0.88,
                error_kw={
                    "ecolor": "black",
                    "elinewidth": 1.35,
                    "capsize": 4.0,
                    "capthick": 1.35,
                    "zorder": 5,
                },
                zorder=3,
            )
            label_y = high + label_offset if median >= 0 else low - label_offset
            axis.text(
                x,
                label_y,
                f"{median:+.1f}",
                ha="center",
                va="bottom" if median >= 0 else "top",
                fontsize=12.5,
                rotation=90,
                color="0.16",
            )

        axis.axhline(0, color="black", linewidth=1.4, zorder=4)
        axis.axvline(divider, color="0.45", linewidth=1.0, linestyle="--")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel("Difference from baseline interval\n($K)")
        axis.grid(axis="y", color="#DEDEDE", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(-0.8, last_position + 0.8)
        axis.set_ylim(full_min - 0.32 * span, full_max + 0.42 * span)

    axes[0].text(
        np.mean(
            [
                positions["EX01_FI_homeowner_outside_SFHA_lower"],
                positions["EX01_BP_wider"],
            ]
        ),
        0.88,
        "HOMEOWNER",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=16.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    axes[0].text(
        np.mean([positions["EX01_FI_renter_lower"], positions["EX01_RL_wider"]]),
        0.88,
        "RENTER",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=16.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )

    tick_ids = [
        f"EX01_{component}_{variation}"
        for component in COMPONENTS
        for variation in VARIATIONS
    ]
    axes[1].set_xticks([positions[case_id] for case_id in tick_ids])
    axes[1].set_xticklabels(
        [str(indexed.loc[case_id, "tested_interval"]) for case_id in tick_ids],
        rotation=54,
        ha="right",
        fontsize=14.2,
    )
    for component in COMPONENTS:
        axes[1].text(
            centers[component],
            -0.41,
            f"{ACTION_LABEL[component]}\nBaseline: {interval_label(*CURRENT[component])}",
            transform=axes[1].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=14.5,
            fontweight="bold",
            color="black",
        )

    figure.subplots_adjust(left=0.105, right=0.99, top=0.92, bottom=0.32, hspace=0.22)
    return figure


def gul_ratio(summary: pd.DataFrame) -> Figure:
    x = np.r_[np.arange(8, dtype=float), np.arange(8, dtype=float) + 10.2]
    centers = {"owner": float(np.mean(x[:8])), "renter": float(np.mean(x[8:]))}
    separator = (x[7] + x[8]) / 2
    with plt.rc_context(
        {
            "font.size": 15,
            "axes.titlesize": 19,
            "axes.labelsize": 16.5,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    ):
        fig, axes = plt.subplots(2, 1, figsize=(15.6, 9.2), sharex=True)
        for ax, name, label in zip(
            axes,
            ["GUL", "actual_loss"],
            [
                "(a) Cumulative GUL difference per household",
                "(b) Cumulative actual loss difference per household",
            ],
        ):
            y = summary[name + "_median"].to_numpy() / 1000
            lo = summary[name + "_q1"].to_numpy() / 1000
            hi = summary[name + "_q3"].to_numpy() / 1000
            assert np.all(lo <= y) and np.all(y <= hi)
            ax.bar(
                x,
                y,
                color=["#3b82f6"] * 8 + ["#22c55e"] * 8,
                edgecolor="black",
                linewidth=0.65,
                alpha=0.88,
                zorder=3,
            )
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                fmt="none",
                ecolor="black",
                elinewidth=1.5,
                capsize=4.5,
                capthick=1.5,
                zorder=5,
            )
            span = max(hi.max() - lo.min(), 1)
            for px, py, pl, ph in zip(x, y, lo, hi):
                offset = max(0.10, span * 0.016)
                ax.text(
                    px,
                    ph + offset if py >= 0 else pl - offset,
                    f"{py:+.1f}",
                    ha="center",
                    va="bottom" if py >= 0 else "top",
                    rotation=90,
                    fontsize=12.5,
                    color=".20",
                )
            ax.set_ylim(min(lo.min(), 0) - 0.15 * span, max(hi.max(), 0) + 0.23 * span)
            ax.set_title(label, loc="left", fontweight="bold")
            ax.set_ylabel(r"Difference from $\theta=0.10$" + "\n($K)")
            ax.axhline(0, color="black", lw=1.4, zorder=4)
            ax.axvline(separator, color=".45", lw=1, ls="--")
            ax.grid(axis="y", color="#E1E1E1", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.8, x[-1] + 0.8)
        for group in ["owner", "renter"]:
            axes[0].text(
                centers[group],
                0.88,
                "HOMEOWNER" if group == "owner" else "RENTER",
                transform=axes[0].get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=16.5,
                fontweight="bold",
                color="black",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 1.5,
                },
            )
        axes[1].set_xticks(x, [f"{t:.2f}" for t in summary.threshold])
        for group in ["owner", "renter"]:
            axes[1].text(
                centers[group],
                -0.22,
                "Tract-level GUL ratio threshold\nBaseline: 0.10",
                transform=axes[1].get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=14.5,
                fontweight="bold",
                color="black",
            )
        fig.subplots_adjust(left=0.105, right=0.99, top=0.92, bottom=0.20, hspace=0.22)
    return fig


def flood_sequence(summary: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.5))
    for ax, g, title in zip(
        axes, ["owner", "renter"], ["(a) Homeowners", "(b) Renters"]
    ):
        for metric, color in COLORS.items():
            for sequence, ls, alpha in [
                ("Observed sequence", "-", 0.16),
                ("Shuffled sequences", "--", 0.08),
            ]:
                s = summary[
                    (summary.group == g)
                    & (summary.metric == metric)
                    & (summary.sequence == sequence)
                ].sort_values("year")
                ax.fill_between(
                    s.year, s.lower, s.upper, color=color, alpha=alpha, lw=0
                )
                ax.plot(s.year, s.center, color=color, ls=ls, lw=2.5)
        ax.set(xlim=(2011, 2023), ylim=(0, None), xlabel="Year", title=title)
        ax.set_xticks(range(2011, 2024, 2))
        ax.grid(alpha=0.2, ls="--")
    axes[0].set_ylabel("Cumulative loss ($K per household)")
    handles = [
        Line2D(
            [],
            [],
            color=c,
            lw=2.5,
            label=m.replace("_", " ").capitalize() if m != "GUL" else m,
        )
        for m, c in COLORS.items()
    ] + [
        Line2D([], [], color=".25", lw=2.5, label="Observed sequence"),
        Line2D([], [], color=".25", ls="--", lw=2.5, label="Shuffled sequences"),
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.03),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


PLOTS = {4: action_intervals, 5: gul_ratio, 6: flood_sequence}
KEYS = {
    4: ["case_id", "component", "tested_interval"],
    5: ["case_id", "group", "threshold"],
    6: ["year", "group", "metric", "sequence"],
}
BAR_VALUES = [
    f"{metric}_{stat}" for metric in COLORS for stat in ("median", "q1", "q3")
]
VALUES = {4: BAR_VALUES, 5: BAR_VALUES, 6: ["center", "lower", "upper"]}


def accepted_csv(number: int) -> Path:
    return DATA / f"FigureS{number}_{REFERENCES[number][0]}.csv"


def check_series(number: int, path: Path) -> pd.DataFrame:
    """Check the plotted data and restore the accepted drawing order."""
    reference_path = accepted_csv(number)
    reference_bytes = reference_path.read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != CSV_SHA256[number]:
        raise ValueError(f"S{number} reference CSV bytes differ: {reference_path}")
    expected = pd.read_csv(BytesIO(reference_bytes))
    data = expected if path == reference_path else pd.read_csv(path)
    keys, values = KEYS[number], VALUES[number]
    if not set(keys + values).issubset(data.columns):
        raise ValueError(f"S{number} plot CSV is missing required columns: {path}")
    indexed = pd.MultiIndex.from_frame(data[keys])
    order = pd.MultiIndex.from_frame(expected[keys])
    if len(indexed) != len(order) or not indexed.is_unique:
        raise ValueError(f"S{number} plot CSV row identities differ: {path}")
    positions = indexed.get_indexer(order)
    if (positions < 0).any():
        raise ValueError(f"S{number} plot CSV row identities differ: {path}")
    data = data.iloc[positions].reset_index(drop=True)
    actual = data[values].to_numpy(dtype=float)
    reference = expected[values].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.allclose(
        actual, reference, atol=1e-9, rtol=0
    ):
        raise ValueError(
            f"S{number} plot values differ from the accepted series: {path}"
        )
    return data


def final_layout(fig: Figure, number: int) -> None:
    """Apply the last typography and spacing edits made to the submission PNGs."""
    for text in fig.findobj(Text):
        text.set_fontsize(text.get_fontsize() * 1.32)
    if number == 4:
        fig.set_size_inches(16.5, 12)
        for axis in fig.axes:
            axis.set_ylabel("Difference from\nbaseline interval\n($K)", fontsize=21)
            axis.title.set_fontsize(24)
            for text in axis.texts:
                if text.get_position()[1] == -0.41:
                    text.set_y(-0.60)
                    text.set_fontsize(17)
        fig.subplots_adjust(left=0.12, right=0.99, top=0.94, bottom=0.29, hspace=0.38)
    elif number == 5:
        fig.tight_layout(h_pad=2.7)
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        legend = fig.legends[0]
        legend.set_ncols(2)
        legend.set_bbox_to_anchor((0.52, 1.04))


def render(number: int, series: pd.DataFrame) -> bytes:
    with matplotlib.rc_context(rc={**matplotlib.rcParamsDefault, **STYLE}):
        fig = PLOTS[number](series)
        try:
            final_layout(fig, number)
            output = BytesIO()
            fig.savefig(
                output,
                format="png",
                dpi=400,
                bbox_inches="tight",
                pad_inches=0.15,
                facecolor="white",
            )
            return output.getvalue()
        finally:
            plt.close(fig)


def render_checked(out_dir: Path, paths: dict[int, Path]) -> None:
    targets = {n: out_dir / f"FigS{n:02d}_{REFERENCES[n][0]}.png" for n in paths}
    for target in targets.values():
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {target}")
    figures = {n: check_series(n, source) for n, source in paths.items()}
    outputs = {}
    for n, series in figures.items():
        img = render(n, series)
        if hashlib.sha256(img).hexdigest() != REFERENCES[n][1]:
            raise ValueError(
                f"S{n} PNG differs from the final archive. "
                "Use the pinned Windows renderer and Times New Roman."
            )
        outputs[n] = img
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, img in outputs.items():
        with targets[n].open("xb") as output:
            output.write(img)
        print(f"PASS {targets[n]}: exact accepted PNG bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--figures", type=int, nargs="+", choices=tuple(REFERENCES))
    for n in REFERENCES:
        parser.add_argument(
            f"--s{n}-series", type=Path, help=f"Verified rebuild CSV for S{n}"
        )
    args = parser.parse_args()
    selected = sorted(set(args.figures or REFERENCES))
    sources = {}
    for n in REFERENCES:
        path = getattr(args, f"s{n}_series")
        if path is not None and n not in selected:
            parser.error(f"--s{n}-series requires selecting Figure S{n}")
        if n in selected:
            sources[n] = path or accepted_csv(n)
    try:
        render_checked(args.figure_dir, sources)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
