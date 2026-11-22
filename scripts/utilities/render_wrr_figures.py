"""Render supported final WRR figures from bundled or rebuilt aggregate series."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


DATA = Path(__file__).resolve().parents[2] / "data" / "figures"
FONT = ["Times New Roman", "Times", "DejaVu Serif"]
SERIES = {
    5: ("Figure5_nfip_validation_corrected_county_labels.csv",
        "Figure5_metrics_corrected_county_labels.csv"),
    6: ("Figure6_cumulative_loss.csv", "Figure6_annual_finance.csv"),
    7: ("Figure7_aep.csv",),
    8: ("Figure8_adaptation_trajectories.csv",),
    9: ("Figure9_tp_by_prone.csv",),
    11: ("Figure11_income_aep.csv",),
}


def trajectory_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "font.family": "serif",
        "font.serif": FONT, "font.size": 14, "axes.titlesize": 19,
        "axes.titleweight": "bold", "axes.labelsize": 18, "xtick.labelsize": 15,
        "ytick.labelsize": 15, "legend.fontsize": 13, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.grid": True, "grid.alpha": 0.12,
        "grid.linestyle": "--",
    })


def nfip_validation(out: Path, series_dir: Path = DATA) -> None:
    # The final series already include the validated county-label correction.
    data = pd.read_csv(series_dir / SERIES[5][0])
    metrics = pd.read_csv(series_dir / SERIES[5][1]).set_index("county")
    counties = {"Essex": "#0072B2", "Morris": "#E69F00", "Passaic": "#009E73"}
    years = list(range(2011, 2024))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": FONT, "font.size": 12,
        "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2, "grid.linestyle": "--",
    })
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.axvspan(2010.5, 2012.5, color="0.5", alpha=0.12, zorder=0)
    x = np.array(years)
    for county, color in counties.items():
        s = data[data.county == county].sort_values("year")
        if s.year.tolist() != years:
            raise ValueError(f"Unexpected years in NFIP diagnostic: {county}")
        ax.fill_between(x, s.q25, s.q75, color=color, alpha=0.10, zorder=1, linewidth=0)
        ax.plot(x, s["median"], color=color, marker="o", ms=6, lw=2.0, zorder=3)
        observed = s.observed_z.to_numpy()
        present = ~np.isnan(observed)
        ax.plot(x[present], observed[present], color=color, marker="s", ms=6, lw=1.8,
                linestyle="--", markerfacecolor="white", markeredgewidth=1.5, zorder=3)
    ax.set_xlabel("Year")
    ax.set_ylabel("z-score (avg. payout per claim)")
    ax.set_xlim(2010.2, 2023.8)
    ax.set_xticks(years)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.5)
    handles = [Patch(facecolor=color, label=f"{county} (r = {metrics.loc[county, 'pearson_r']:.2f})")
               for county, color in counties.items()]
    styles = [
        Line2D([0], [0], color="0.3", marker="o", lw=2.0, label="Simulated (median)"),
        Line2D([0], [0], color="0.3", marker="s", lw=1.8, linestyle="--",
               markerfacecolor="white", markeredgewidth=1.5, label="Actual (NFIP)"),
        Patch(facecolor="0.5", alpha=0.12, label="Warm-up"),
    ]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, 1.0),
                    fontsize=10.5, frameon=True, framealpha=0.95, ncol=1)
    ax.add_artist(leg)
    ax.legend(handles=styles, loc="upper right", fontsize=10.5, frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def adoption(out: Path, series_dir: Path = DATA) -> None:
    data = pd.read_csv(series_dir / SERIES[8][0])
    trajectory_style()
    specs = (
        ("homeowner", "FI", "#1a6faf", "o", "Flood Insurance (FI) — Homeowner"),
        ("renter", "FI", "#d62728", "s", "Flood Insurance (FI) — Renter"),
        ("homeowner", "EH", "#2ca02c", "o", "Elevation (EH) — Homeowner"),
        ("renter", "RL", "#d62728", "o", "Relocation (RL) — Renters without FI"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (ax, spec) in enumerate(zip(axes.ravel(), specs, strict=True)):
        group, action, color, marker, title = spec
        for year in (2011, 2021):
            ax.axvspan(year - 0.45, year + 0.45, color="#dceeff", alpha=0.72, zorder=0)
        series = {}
        for exposure, ls, open_marker in (("flood-prone", "-", False), ("non-prone", "--", True)):
            s = data[(data["group"] == group) & (data["action"] == action) &
                     (data["exposure"] == exposure)].sort_values("year")
            series[exposure] = s
            x, y = s["year"].to_numpy(int), s["median"].to_numpy(float) * 100
            ax.fill_between(x, s["q25"] * 100, s["q75"] * 100, color=color,
                            alpha=0.24 if not open_marker else 0.11, linewidth=0)
            ax.plot(x, y, color=color, ls=ls, marker=marker, ms=5, lw=2,
                    markerfacecolor="none" if open_marker else color, markeredgewidth=1.3,
                    label=(f"{action} ({exposure})" if action != "RL" else
                           f"{exposure.capitalize()} (renters without FI)"))
        ymax = (60, 60, 6, 30)[idx]
        for pos in (0, -1):
            values = {k: s["median"].to_numpy(float)[pos] * 100 for k, s in series.items()}
            visible = ["flood-prone"] if idx == 2 and pos == -1 else list(series)
            high = max(values[k] for k in visible)
            for exposure in visible:
                value = values[exposure]
                year = int(series[exposure]["year"].iloc[pos])
                offset = 12 if value == high or value <= 0.04 * ymax else -16
                if idx == 0 and pos == -1 and exposure == "flood-prone":
                    offset = -16
                ax.annotate(f"{value:.1f}%", xy=(year, value), xytext=(5 if pos == 0 else -5, offset),
                            textcoords="offset points", color=color, fontsize=12, fontweight="bold",
                            ha="center", va="bottom" if offset > 0 else "top",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.7))
        fp = series["flood-prone"]["median"].to_numpy(float) * 100
        npn = series["non-prone"]["median"].to_numpy(float) * 100
        for pos, dx in ((0, 0.55), (-1, -0.55)):
            year = int(series["flood-prone"]["year"].iloc[pos])
            if idx == 2 and pos == -1 and abs(fp[pos] - npn[pos]) < 0.5:
                continue
            ax.annotate("", xy=(year + dx, fp[pos]), xytext=(year + dx, npn[pos]),
                        arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.4))
            ax.text(year + (1.35 if pos == 0 else -1.35), (fp[pos] + npn[pos]) / 2,
                    f"Δ={abs(fp[pos] - npn[pos]):.1f} pp", fontsize=10.5,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.4"))
        ax.set(title=title, ylabel="Adoption rate (%)", xlabel="Year")
        ax.set_xticks(range(2011, 2024, 2))
        ax.set_xlim(2010.3, 2023.7)
        ax.set_ylim(0, ymax)
        ax.set_yticks(np.arange(0, ymax + 1, (10, 10, 1, 5)[idx]))
        ax.legend(loc="upper right", fontsize=14, framealpha=0.96,
                  borderpad=0.6, labelspacing=0.55, handlelength=2.2)
        ax.text(-0.10, 1.04, f"({chr(97 + idx)})", transform=ax.transAxes,
                fontsize=18, fontweight="bold", ha="left", va="bottom")
    fig.tight_layout(h_pad=2.6, w_pad=2.5)
    # Retain the final legend-only change after the panel geometry is set.
    for ax in fig.axes:
        for text in ax.get_legend().get_texts():
            text.set_fontsize(16)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)


def threat_perception(out: Path, series_dir: Path = DATA) -> None:
    data = pd.read_csv(series_dir / SERIES[9][0])
    trajectory_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (ax, (group, color)) in enumerate(zip(
        axes, (("homeowner", "#3b82f6"), ("renter", "#22c55e")), strict=True
    )):
        for year in (2011, 2021):
            ax.axvspan(year - 0.45, year + 0.45, color="#dceeff", alpha=0.72, zorder=0)
        series = {}
        for exposure, ls, marker, open_marker in (
            ("flood-prone", "-", "o", False), ("non-prone", "--", "s", True)
        ):
            s = data[(data["group"] == group) & (data["exposure"] == exposure)].sort_values("year")
            series[exposure] = s
            x, y = s["year"].to_numpy(int), s["median"].to_numpy(float)
            ax.fill_between(x, s["q25"], s["q75"], color=color,
                            alpha=0.22 if not open_marker else 0.10, linewidth=0)
            ax.plot(x, y, color=color, ls=ls, marker=marker, ms=5, lw=2,
                    markerfacecolor="none" if open_marker else color,
                    markeredgewidth=1.3, label=exposure.capitalize())
            for pos in (0, -1):
                ax.text(x[pos], y[pos] + (0.035 if exposure == "flood-prone" else -0.06),
                        f"{y[pos]:.2f}", color=color, fontsize=11, fontweight="bold",
                        ha="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.7))
        fp = series["flood-prone"]["median"].to_numpy(float)
        npn = series["non-prone"]["median"].to_numpy(float)
        for pos, dx in ((0, 0.5), (-1, -0.5)):
            year = int(series["flood-prone"]["year"].iloc[pos])
            ax.annotate("", xy=(year + dx, fp[pos]), xytext=(year + dx, npn[pos]),
                        arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.4))
            ax.text(year + (1.35 if pos == 0 else -1.35), (fp[pos] + npn[pos]) / 2,
                    f"Δ={abs(round(fp[pos], 2) - round(npn[pos], 2)):.2f}", fontsize=11,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.4"))
        ax.set_title(group.capitalize())
        ax.set_ylabel("Mean TP")
        ax.set_xlabel("Year")
        ax.set_xticks(list(range(2011, 2024, 2)))
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", framealpha=0.96)
        ax.text(-0.10, 1.04, f"({chr(97 + idx)})", transform=ax.transAxes,
                fontsize=18, fontweight="bold", ha="left", va="bottom")
    fig.tight_layout(w_pad=2.5)
    for ax in axes:
        for text in ax.get_legend().get_texts():
            text.set_fontsize(17)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)


def income(out: Path, series_dir: Path = DATA) -> None:
    data = pd.read_csv(series_dir / SERIES[11][0])
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 14, "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "axes.titlesize": 17, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    colors = {"GUL": "#d62728", "Actual loss": "#0072B2"}
    fig, ax = plt.subplots(figsize=(10, 6.6))
    endpoints = {}
    for group, ls in (("homeowner", "-"), ("renter", "--")):
        endpoints[group] = {}
        for metric, color in colors.items():
            s = data[(data.group == group) & (data.metric == metric)].sort_values("aep", ascending=False)
            if len(s) != 600 or not s.value.is_monotonic_increasing:
                raise ValueError(f"Unexpected AEP series: {group}/{metric}")
            ax.plot(s.aep, s.value, color=color, linestyle=ls, lw=2.5)
            endpoints[group][metric] = float(s.value.iloc[-1])
    ax.set(xlim=(1, 0), ylim=(0, float(data.value.max()) * 1.10),
           xlabel="Aggregate exceedance probability",
           ylabel="Annual household loss (% of median household income)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    labels = {}
    for group, pos in (("homeowner", (.29, 27)), ("renter", (.32, 5.0))):
        v = endpoints[group]
        gap, x = v["GUL"] - v["Actual loss"], .012
        ax.annotate("", xy=(x, v["GUL"]), xytext=(x, v["Actual loss"]),
                    arrowprops=dict(arrowstyle="<->", color=".25", lw=1.4))
        text = ax.annotate(f"{group.capitalize()} gap ≈ {gap:.1f}%",
                           xy=(x, (v["GUL"] + v["Actual loss"]) / 2), xytext=pos,
                           fontsize=13, fontweight="bold", ha="center", va="center", color=".2",
                           bbox=dict(boxstyle="round,pad=.3", fc="white", ec=".5", lw=.6),
                           arrowprops=dict(arrowstyle="-", color=".5", lw=.6))
        labels[group] = (text, gap)
    handles = [Line2D([], [], color=colors[m], lw=2.5, label=m) for m in colors]
    handles += [Line2D([], [], color=".25", lw=2.5, label="Homeowner"),
                Line2D([], [], color=".25", ls="--", lw=2.5, label="Renter")]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=13)
    ax.grid(alpha=.2, ls="--")
    fig.tight_layout()
    for group, (text, gap) in labels.items():
        text.set_text(f"{group.capitalize()} protection margin\n≈ {gap:.1f}% of median income")
        if group == "renter":
            text.set_position((text.get_position()[0], 4.4))
    fig.savefig(out, format="png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--series-dir", type=Path, default=DATA,
                        help="Read all selected figure inputs here (default: bundled data/figures).")
    parser.add_argument("--font-dir", type=Path, help="Optional Times New Roman directory for Figures 6 and 7")
    figures = {5: nfip_validation, 8: adoption, 9: threat_perception, 11: income}
    parser.add_argument("--figures", type=int, nargs="+", choices=tuple(SERIES), default=[5, 8, 11],
                        help="Figures 6 and 7 need the vector dependencies; Figure 9 uses its separate environment.")
    args = parser.parse_args()
    if 9 in args.figures and set(args.figures) != {9}:
        parser.error("Render Figure 9 separately with the environment in environment-figure9-win64.lock.")
    outputs = {n: args.figure_dir / f"Figure_{n:02}.png" for n in sorted(set(args.figures))}
    missing = [args.series_dir / name for n in outputs for name in SERIES[n]
               if not (args.series_dir / name).is_file()]
    if missing:
        parser.error("Missing input series: " + ", ".join(map(str, missing)))
    for path in outputs.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}")
    verified = {}
    if 6 in outputs:
        try:
            if __package__:
                from .render_wrr_sq1_png import verified_png
            else:
                from render_wrr_sq1_png import verified_png
            verified[6] = verified_png(args.series_dir, args.font_dir)
        except (ImportError, KeyError, OSError, ValueError) as error:
            parser.exit(1, f"Figure 6: {error}\n")
    if 7 in outputs:
        if __package__:
            from .render_wrr_aep_png import verified_png
        else:
            from render_wrr_aep_png import verified_png
        try:
            verified[7] = verified_png(None if args.series_dir == DATA else args.series_dir, args.font_dir)
        except (ImportError, KeyError, OSError, ValueError) as error:
            parser.exit(1, f"Figure 7: {error}\n")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    for number, path in outputs.items():
        if number in verified:
            with path.open("xb") as png:
                png.write(verified[number])
        else:
            with plt.rc_context():
                figures[number](path, args.series_dir)
        print(path)


if __name__ == "__main__":
    main()
