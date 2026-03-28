# -*- coding: utf-8 -*-
"""
Fig 6 Combined: Cumulative Composition (stock) + Annual Adoption Rate (flow)
Layout: 2×2
  (a) Homeowner — Cumulative Composition (stacked bar)
  (b) Renter — Cumulative Composition (stacked bar)
  (c) Homeowner — Annual Adoption Rate (mean lines)
  (d) Renter — Annual Adoption Rate (mean lines)
"""
import sys, re, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils.plots_modular.style import set_paper_style, panel_label, shade_severe_years

set_paper_style()

BASE = ROOT / "outputs" / "baseline" / "baseline"
SEVERE = [2011, 2021]

# ── Colors ──
C = {
    "Both (EH+FI)": "#4f46e5",  # indigo
    "EH": "#059669",             # green
    "FI": "#60a5fa",             # light blue
    "BP": "#f59e0b",             # orange
    "DN": "#9ca3af",             # gray
    "RL": "#dc2626",             # red
}
# Line colors for annual rates
LC = {
    "FI": "#1f77b4",
    "EH": "#2ca02c",
    "BP": "#f59e0b",
    "RL": "#dc2626",
    "DN": "#9ca3af",
}
OWNER_FI_COLOR = "#2c3e50"
RENTER_FI_COLOR = "#d62828"

# ── Load decisions ──
print("Loading decision data...")
file_map = {}
for f in sorted(glob.glob(str(BASE / "decisions/decisions_mgmix_*.csv"))):
    m = re.search(r"mgmix_(\d{4})\.csv", f)
    if m:
        file_map[int(m.group(1))] = f

years = sorted(file_map.keys())
x = np.arange(len(years))

# ── Compute STOCK (cumulative composition) ──
set_EH = set()
set_BP = set()
owner_stats = []
renter_stats = []

for i, y in enumerate(years):
    df = pd.read_csv(file_map[y])

    # Owners
    owners = df[df["group"] == "owner"]
    current_ehs = set(owners[owners["action"] == "EH"]["i"])
    if "ELEV_FT" in owners.columns:
        current_ehs.update(set(owners[owners["ELEV_FT"] > 0]["i"]))
    set_EH.update(current_ehs)
    current_bps = set(owners[owners["action"] == "BP"]["i"])
    set_BP.update(current_bps)
    n_bp_current = len(current_bps)

    stayers = owners[owners["action"] != "BP"]
    is_fi_mask = (owners["action"] == "FI")
    if "POLICY_NAME" in owners.columns:
        has_pol = owners["POLICY_NAME"].notna() & (owners["POLICY_NAME"] != "")
        is_fi_mask = is_fi_mask | has_pol

    is_eh_status = stayers["i"].isin(set_EH)
    is_fi_status = is_fi_mask[stayers.index]

    owner_stats.append({
        "year": y,
        "Both (EH+FI)": (is_eh_status & is_fi_status).sum(),
        "EH": (is_eh_status & ~is_fi_status).sum(),
        "FI": (~is_eh_status & is_fi_status).sum(),
        "BP": n_bp_current,
        "DN": (~is_eh_status & ~is_fi_status).sum(),
    })

    # Renters
    renters = df[df["group"] == "renter"]
    r_fi = (renters["action"] == "FI").sum()
    if "POLICY_NAME" in renters.columns:
        has_pol = renters["POLICY_NAME"].notna() & (renters["POLICY_NAME"] != "")
        r_fi = ((renters["action"] == "FI") | has_pol).sum()
    r_rl = (renters["action"] == "RL").sum()
    r_dn = max(0, len(renters) - r_fi - r_rl)
    renter_stats.append({"year": y, "FI": r_fi, "RL": r_rl, "DN": r_dn})

df_own = pd.DataFrame(owner_stats)
df_rent = pd.DataFrame(renter_stats)

# ── Compute FLOW (annual adoption rates, population-level) ──
owner_flow = {"FI": [], "EH": [], "BP": [], "DN": []}
renter_flow = {"FI": [], "RL": [], "DN": []}

for y in years:
    df = pd.read_csv(file_map[y])

    ow = df[df["group"] == "owner"]
    no = len(ow)
    ofi = ((ow["action"] == "FI") | (ow["POLICY_NAME"].notna() & (ow["POLICY_NAME"].astype(str).str.len() > 0))).sum() / no if no > 0 else 0
    oeh = ((ow["action"] == "EH") | (ow.get("ELEV_FT", pd.Series([0]*no)) > 0)).sum() / no if no > 0 else 0
    obp = (ow["action"] == "BP").sum() / no if no > 0 else 0
    odn = (ow["action"] == "DN").sum() / no if no > 0 else 0
    owner_flow["FI"].append(ofi)
    owner_flow["EH"].append(oeh)
    owner_flow["BP"].append(obp)
    owner_flow["DN"].append(odn)

    rt = df[df["group"] == "renter"]
    nr = len(rt)
    rfi = ((rt["action"] == "FI") | (rt["POLICY_NAME"].notna() & (rt["POLICY_NAME"].astype(str).str.len() > 0))).sum() / nr if nr > 0 else 0
    rrl = (rt["action"] == "RL").sum() / nr if nr > 0 else 0
    rdn = (rt["action"] == "DN").sum() / nr if nr > 0 else 0
    renter_flow["FI"].append(rfi)
    renter_flow["RL"].append(rrl)
    renter_flow["DN"].append(rdn)

# ── Plot 3×2 — Left=Homeowner, Right=Renter ──
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
ax_a, ax_b = axes[0]  # stock: Homeowner | Renter
ax_c, ax_d = axes[1]  # FI:    Homeowner | Renter
ax_e, ax_f = axes[2]  # EH (Homeowner) | RL (Renter)

owner_actions = ["Both (EH+FI)", "EH", "FI", "BP", "DN"]
renter_actions = ["FI", "RL", "DN"]


def draw_stacked(ax, stats_df, acts):
    totals = stats_df[list(acts)].sum(axis=1).values
    bottom = np.zeros(len(stats_df))
    for a in acts:
        if a not in stats_df.columns:
            continue
        vals = stats_df[a].values
        heights = np.divide(vals, totals, out=np.zeros_like(totals, dtype=float),
                            where=totals != 0)
        ax.bar(x, heights, bottom=bottom, width=0.75, color=C.get(a, "#9ca3af"),
               edgecolor="white", linewidth=0.5, alpha=0.9, label=a)
        bottom += heights
    ax.set_ylabel("Household composition (%)", fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="both", labelsize=11)


def annotate_bar(ax, stats_df, acts, idx):
    """Add percentage labels inside bar at given index."""
    total = stats_df[list(acts)].iloc[idx].sum()
    if total == 0:
        return
    bottom = 0.0
    bar_x = 0 if idx == 0 else len(stats_df) - 1
    for a in acts:
        if a not in stats_df.columns:
            continue
        val = stats_df[a].iloc[idx]
        height = val / total if total > 0 else 0
        if height >= 0.03:
            mid = bottom + height / 2
            pct = f"{height*100:.0f}%"
            ax.text(bar_x, mid, pct, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#222222",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.7))
        bottom += height


def annotate_endpoint(ax, x_pos, val, color, offset_y=0.01, va="bottom"):
    fmt = f"{val*100:.1f}%" if val < 0.05 else f"{val*100:.0f}%"
    ax.annotate(fmt, xy=(x_pos, val),
                xytext=(x_pos, val + offset_y),
                fontsize=10, fontweight="bold", color=color, ha="center", va=va,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, linewidth=0.6, alpha=1.0))


def setup_flow_ax(ax, title, plbl):
    """Common setup for flow panels."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    panel_label(ax, plbl)
    ax.set_ylabel("Weighted tract-level adoption rate (%)", fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, None)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(y) for y in years[::2]])
    ax.tick_params(axis="both", labelsize=11)


# (a) Homeowner — Cumulative Composition
draw_stacked(ax_a, df_own, owner_actions)
annotate_bar(ax_a, df_own, owner_actions, 0)
annotate_bar(ax_a, df_own, owner_actions, -1)
ax_a.set_title("Homeowner", fontsize=15, fontweight="bold", pad=10)
panel_label(ax_a, "(a)")
handles_a, labels_a = ax_a.get_legend_handles_labels()
ax_a.legend(handles_a, labels_a, loc="upper center", fontsize=10, frameon=True,
            facecolor="white", edgecolor="0.6", framealpha=0.95, ncol=len(labels_a))
ax_a.set_xticks(x[::2])
ax_a.set_xticklabels([str(y) for y in years[::2]])
ax_a.set_xlabel("Year", fontsize=12)

# (b) Renter — Cumulative Composition
draw_stacked(ax_b, df_rent, renter_actions)
annotate_bar(ax_b, df_rent, renter_actions, 0)
annotate_bar(ax_b, df_rent, renter_actions, -1)
ax_b.set_title("Renter", fontsize=15, fontweight="bold", pad=10)
panel_label(ax_b, "(b)")
handles_b, labels_b = ax_b.get_legend_handles_labels()
ax_b.legend(handles_b, labels_b, loc="upper center", fontsize=10, frameon=True,
            facecolor="white", edgecolor="0.6", framealpha=0.95, ncol=len(labels_b))
ax_b.set_xticks(x[::2])
ax_b.set_xticklabels([str(y) for y in years[::2]])
ax_b.set_xlabel("Year", fontsize=12)

# ── Classify tracts: flood-prone vs non-prone ──
import json
_flood_cfg = Path(ROOT / "config" / "overall_md_mean_by_tract_2011_2023.json")
with open(_flood_cfg, encoding="utf-8") as _f:
    _flood_data = json.load(_f)
PRONE_TRACTS = set()
NON_PRONE_TRACTS = set()
for _t in _flood_data:
    _vals = [_t[k] for k in _t if k.endswith("_mean")]
    if max(_vals) > 0.05:
        PRONE_TRACTS.add(_t["CensusTract"])
    else:
        NON_PRONE_TRACTS.add(_t["CensusTract"])
print(f"Flood-prone tracts: {len(PRONE_TRACTS)}, Non-prone: {len(NON_PRONE_TRACTS)}")

# ── Compute tract-level rates for weighted stats ──
def compute_tract_rates(group, action):
    """Return {year: {tract: rate}} and {year: {tract: n_hh}}."""
    yr_rates, yr_pop = {}, {}
    for y in years:
        df = pd.read_csv(file_map[y])
        grp = df[df["group"] == group].copy()
        grp["tract_geoid"] = grp["tract_geoid"].astype(str)
        rates, pops = {}, {}
        for t in sorted(grp["tract_geoid"].unique()):
            t_df = grp[grp["tract_geoid"] == t]
            n = len(t_df)
            if n == 0:
                continue
            pops[t] = n
            if action == "FI":
                is_act = (t_df["action"] == "FI")
                if "POLICY_NAME" in t_df.columns:
                    has_pol = t_df["POLICY_NAME"].notna() & (t_df["POLICY_NAME"].astype(str).str.len() > 0)
                    is_act = is_act | has_pol
                rates[t] = is_act.sum() / n
            elif action == "EH":
                is_act = (t_df["action"] == "EH")
                if "ELEV_FT" in t_df.columns:
                    is_act = is_act | (t_df["ELEV_FT"] > 0)
                rates[t] = is_act.sum() / n
            else:
                rates[t] = (t_df["action"] == action).sum() / n
        yr_rates[y] = rates
        yr_pop[y] = pops
    return yr_rates, yr_pop

def get_weighted_mean_by_exposure(yr_rates, yr_pop, tract_set):
    """Population-weighted mean for a subset of tracts."""
    means = []
    for y in years:
        tracts = [t for t in yr_rates[y] if t in tract_set]
        if not tracts:
            means.append(0.0)
            continue
        v = np.array([yr_rates[y][t] for t in tracts])
        w = np.array([yr_pop[y].get(t, 1) for t in tracts], dtype=float)
        w = np.where(w > 0, w, 1e-6)
        means.append(np.average(v, weights=w))
    return np.array(means)

def plot_action_two_lines(ax, group, action, color, label, marker_prone="o", marker_non="s", ms=5):
    """Plot two lines: solid for flood-prone, dashed for non-prone."""
    yr_r, yr_p = compute_tract_rates(group, action)
    prone_mean = get_weighted_mean_by_exposure(yr_r, yr_p, PRONE_TRACTS)
    non_mean = get_weighted_mean_by_exposure(yr_r, yr_p, NON_PRONE_TRACTS)
    ax.plot(x, prone_mean, f"-{marker_prone}", ms=ms, lw=2.2, color=color,
            label=f"{label} (flood-prone)", zorder=3)
    ax.plot(x, non_mean, f"--{marker_non}", ms=ms-1, lw=1.5, color=color,
            alpha=0.6, label=f"{label} (non-prone)", zorder=2)
    return prone_mean, non_mean

# ── (c) Homeowner FI (prone vs non-prone) ──
shade_severe_years(ax_c, years, SEVERE)
o_fi_p, o_fi_n = plot_action_two_lines(ax_c, "owner", "FI", "#1f77b4", "FI", "o", "o")
# Prone: above
annotate_endpoint(ax_c, x[0],  o_fi_p[0],  "#1f77b4", offset_y=0.03)
annotate_endpoint(ax_c, x[-1], o_fi_p[-1], "#1f77b4", offset_y=0.03)
# Non-prone: above (well below prone, no overlap)
annotate_endpoint(ax_c, x[0],  o_fi_n[0],  "#1f77b4", offset_y=0.03)
annotate_endpoint(ax_c, x[-1], o_fi_n[-1], "#1f77b4", offset_y=0.03)
setup_flow_ax(ax_c, "Flood Insurance (FI) — Homeowner", "(c)")
ax_c.set_ylim(0, 1.0)
ax_c.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white",
            edgecolor="0.6", framealpha=0.95)

# ── (d) Renter FI (prone vs non-prone) ──
shade_severe_years(ax_d, years, SEVERE)
r_fi_p, r_fi_n = plot_action_two_lines(ax_d, "renter", "FI", "#d62728", "FI", "s", "s")
# Prone: above
annotate_endpoint(ax_d, x[0],  r_fi_p[0],  "#d62728", offset_y=0.03)
annotate_endpoint(ax_d, x[-1], r_fi_p[-1], "#d62728", offset_y=0.03)
# Non-prone: above (much lower than prone, no overlap)
annotate_endpoint(ax_d, x[0],  r_fi_n[0],  "#d62728", offset_y=0.03)
annotate_endpoint(ax_d, x[-1], r_fi_n[-1], "#d62728", offset_y=0.03)
setup_flow_ax(ax_d, "Flood Insurance (FI) — Renter", "(d)")
ax_d.set_ylim(0, 1.0)
ax_d.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white",
            edgecolor="0.6", framealpha=0.95)

# ── (e) EH — Homeowner (prone vs non-prone) ──
shade_severe_years(ax_e, years, SEVERE)
o_eh_p, o_eh_n = plot_action_two_lines(ax_e, "owner", "EH", LC["EH"], "EH", "o", "o")
# Prone start: below (non-prone starts slightly higher, so separate them)
annotate_endpoint(ax_e, x[0],  o_eh_p[0],  LC["EH"], offset_y=-0.012, va="top")
# Non-prone start: above
annotate_endpoint(ax_e, x[0],  o_eh_n[0],  LC["EH"], offset_y=0.012)
# End: only prone (both converge to ~0.1%)
annotate_endpoint(ax_e, x[-1], o_eh_p[-1], LC["EH"], offset_y=0.006)
setup_flow_ax(ax_e, "Elevation (EH) — Homeowner", "(e)")
ax_e.set_ylim(0, 0.072)  # headroom for "4.5%" label above non-prone start
ax_e.legend(loc="center right", fontsize=10, frameon=True, facecolor="white",
            edgecolor="0.6", framealpha=0.95)

# ── (f) RL — Renter (prone vs non-prone) ──
shade_severe_years(ax_f, years, SEVERE)
r_rl_p, r_rl_n = plot_action_two_lines(ax_f, "renter", "RL", LC["RL"], "RL", "o", "o")
# Prone (lower line): above
annotate_endpoint(ax_f, x[0],  r_rl_p[0],  LC["RL"], offset_y=0.018)
annotate_endpoint(ax_f, x[-1], r_rl_p[-1], LC["RL"], offset_y=0.018)
# Non-prone (upper line): above
annotate_endpoint(ax_f, x[0],  r_rl_n[0],  LC["RL"], offset_y=0.018)
annotate_endpoint(ax_f, x[-1], r_rl_n[-1], LC["RL"], offset_y=0.018)
setup_flow_ax(ax_f, "Relocation (RL) — Renter", "(f)")
ax_f.legend(loc="lower left", fontsize=10, frameon=True, facecolor="white",
            edgecolor="0.6", framealpha=0.95)

fig.tight_layout(h_pad=3.0, w_pad=2.5)

# Save
out_dir = BASE / "visualization" / "decisions"
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ["png", "pdf"]:
    fig.savefig(out_dir / f"fig6_combined_stock_flow.{ext}", dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_dir / 'fig6_combined_stock_flow.png'}")
