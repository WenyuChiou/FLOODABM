from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from matplotlib.colors import Normalize


from .actions_core import (
    selection_vectors as vectors_for_selections,  # ← 用別名對齊下方呼叫
    heatmap_matrix, eh_coverage_series, read_action_share_all
)

# ---- style helpers ----
def _set_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.titleweight": "bold", # Bold
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "CMU Serif"],
    })

def _panel_label(ax, label="(a)", x=-0.12, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold",
            ha="left", va="bottom")


def _ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

# 以 scenario 統一配色（可自行調整）
SCN_COLORS = {
    "RT=0.1": "#4C78A8",
    "RT=0.2": "#F58518",
    "RT=0.3": "#54A24B",
    "RT=0.4": "#E45756",
    "RT=0.5": "#B279A2",
}

# =============== 1) Actions：依 scenario 著色的 grouped boxplot ===============
def plot_actions_grouped_boxplot(run_dirs_map: Dict[str, Path],
                                 selections: List[Tuple[str, Optional[str], str]],
                                 pattern: str,
                                 out_png: Path,
                                 title: str = "Tract-level action shares across scenarios (avg over years)",
                                 figsize: Tuple[int,int] = (18,7)) -> None:
    """
    每個 action 一個群組；群內用不同顏色的 box 代表不同 scenario。
    selections 例如：
      [("Homeowner(FI)","Owner","FI"), ("Renter(FI)","Renter","FI"),
       ("BP",None,"BP"), ("RL",None,"RL"), ("DN",None,"DN")]
    """

    _set_style()

    scenarios = list(run_dirs_map.keys())
    K = len(selections)
    M = len(scenarios)
    x = np.arange(K, dtype=float)

    # 收資料（注意把 pattern 傳下去）
    vec_by_scn: Dict[str, Dict[str, np.ndarray]] = {}
    for scn, rdir in run_dirs_map.items():
        vec_by_scn[scn] = vectors_for_selections(rdir, selections, pattern)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    group_width = 0.8
    box_w = group_width / max(1, M)
    offsets = np.linspace(-group_width/2 + box_w/2, group_width/2 - box_w/2, M)

    # 若有你預設的配色 dict，就沿用；否則讓 matplotlib 自選
    SCN_COLORS = {
        "RT=0.1": "#4C78A8", "RT=0.2": "#F58518", "RT=0.3": "#54A24B",
        "RT=0.4": "#E45756", "RT=0.5": "#B279A2",
    }

    legend_handles = []
    for j, scn in enumerate(scenarios):
        color = SCN_COLORS.get(scn, None)
        xs = x + offsets[j]
        data = [vec_by_scn[scn][lab] for (lab, _, _) in selections]

        bp = ax.boxplot(
            data, positions=xs, widths=box_w*0.9,
            showfliers=False, patch_artist=True, whis=(5, 95)
        )
        for patch in bp["boxes"]:
            if color:
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        for key in ["whiskers","caps","medians"]:
            for ln in bp[key]:
                ln.set_color("black")
                ln.set_linewidth(0.8)

        h = plt.Line2D([0],[0], marker="s", linestyle="",
                       markerfacecolor=(color if color else "grey"),
                       markeredgecolor="black", alpha=0.55,
                       markersize=12, label=scn)
        legend_handles.append(h)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for (lab,_,_) in selections], fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share", fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(handles=legend_handles, title="Scenario", loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =============== 2) Actions：依 MG/NMG 著色的 heatmap ===============
def _normalize_run_dir(run_dir: Path) -> Path:
    """支援 outputs/experiments/EXP/baseline/* 與 EXP/*"""
    run_dir = Path(run_dir)
    b = run_dir / "baseline"
    if (b / "finance").exists() or (b / "vulnerability").exists():
        return b
    return run_dir



import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _share_by_people_mean(run_dir: Path, group: Optional[str], action: str) -> float:
    """
    以「採取 action 的戶數 / 當年該群組戶數」為 share，先算每年，再跨年取平均。
    - group: "Owner" | "Renter" | None（None 會依 action 自動判斷：RL->Renter、BP->Owner、DN->兩群都可→需搭配明確 group）
    - action: "FI" | "EH" | "BP" | "RL" | "DN"
    回傳單一 scalar（該 run 的平均 share）。
    """
    run_dir = Path(run_dir)
    dec_dir = run_dir / "decisions"
    parts = sorted(dec_dir.glob("decisions_mgmix_*.csv"))
    if not parts:
        return float("nan")

    gnorm = (group or "").strip().lower()
    if gnorm == "homeowner": gnorm = "owner"
    a = action.strip().upper()

    vals = []
    years = []
    for p in parts:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "group" not in df.columns or "action" not in df.columns:
            continue
        df["group"] = df["group"].astype(str).str.lower()
        df["action"] = df["action"].astype(str).str.upper()

        # 若 group 未指定，依 action 做合理預設
        if not gnorm:
            if a == "RL":
                gnorm = "renter"
            elif a == "BP":
                gnorm = "owner"
            else:
                # FI / DN / EH 等需要明確群組，若沒給就整體分母→不建議
                pass

        sub = df.copy()
        if gnorm:
            sub = sub[sub["group"].eq(gnorm)]
        if sub.empty:
            continue

        denom = len(sub)
        numer = int((sub["action"] == a).sum())
        share = numer / denom if denom > 0 else np.nan
        vals.append(share)

        # 取年（檔名帶年或欄位 year）
        if "year" in sub.columns:
            years.append(int(pd.to_numeric(sub["year"], errors="coerce").dropna().unique()[0]))
        else:
            years.append(None)

    v = np.array(vals, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def plot_actions_heatmaps_mg_nmg(
    run_dirs_map: Dict[str, Path],
    selections: List[Tuple[str, Optional[str], str]],
    mg_values: List[float],
    nmg_values: List[float],
    pattern: str,
    out_png: Path,
    digits: int = 3,         # ✅ 改為小數點後三位
    per_panel_range: bool = True,
    pad: float = 0.02,
    cmaps: Optional[List[str]] = None,
) -> None:
    """
    2×3 子圖（含 Owner(DN) 與 Renter(DN)），每個子圖都有自己的 colorbar。
    每格值 = 以『採取 action 的戶數 / 當年群組戶數』為 share，跨年取平均。
    """
    _set_style()

    # ✅ 確保 DN 分群共六張
    want_order = ["Homeowner(FI)", "Renter(FI)", "BP", "RL", "Homeowner(DN)", "Renter(DN)"]
    sel_map = {s[0]: s for s in selections}
    sels = [sel_map[k] for k in want_order if k in sel_map]

    if cmaps is None:
        cmaps = ["Blues", "Greens", "Purples", "Oranges", "Greys", "Greens"]
    assert len(sels) <= len(cmaps), "colormap 列表不足，請傳入更長的 cmaps"

    mg_list = sorted({float(x) for x in mg_values})
    nmg_list = sorted({float(x) for x in nmg_values})
    M, N = len(nmg_list), len(mg_list)

    def _edges(vals: List[float]) -> np.ndarray:
        vals = np.array(sorted(vals), dtype=float)
        if vals.size == 1:
            step = 0.05
            return np.array([vals[0]-step, vals[0]+step], dtype=float)
        mids = (vals[:-1] + vals[1:]) / 2.0
        first = vals[0] - (mids[0] - vals[0])
        last = vals[-1] + (vals[-1] - mids[-1])
        return np.r_[first, mids, last]

    mg_edges = _edges(mg_list)
    nmg_edges = _edges(nmg_list)

    fig = plt.figure(figsize=(15.5, 9.0), dpi=300, constrained_layout=False)
    gs = fig.add_gridspec(2, 3, wspace=0.24, hspace=0.34)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    def _find_run(mg: float, nmg: float) -> Optional[Path]:
        key = f"MG={mg:g},NMG={nmg:g}"
        return run_dirs_map.get(key, None)

    for j, sel in enumerate(sels):
        lab, grp, act = sel
        ax = axes[j]
        mat = np.full((M, N), np.nan, dtype=float)

        for r_i, nmg in enumerate(nmg_list):
            for c_i, mg in enumerate(mg_list):
                rdir = _find_run(mg, nmg)
                if rdir is None:
                    continue
                try:
                    mat[r_i, c_i] = _share_by_people_mean(rdir, grp, act)
                except Exception:
                    mat[r_i, c_i] = np.nan

        # ✅ 自動縮放色階
        if per_panel_range:
            finite = mat[np.isfinite(mat)]
            if finite.size:
                lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
                if np.isclose(lo, hi):
                    lo = max(0.0, lo - 0.02)
                    hi = min(1.0, hi + 0.02)
                rng = hi - lo
                lo = max(0.0, lo - pad * rng)
                hi = min(1.0, hi + pad * rng)
            else:
                lo, hi = 0.0, 1.0
            norm = Normalize(vmin=lo, vmax=hi, clip=True)
        else:
            norm = Normalize(vmin=0.0, vmax=1.0, clip=True)

        quad = ax.pcolormesh(
            mg_edges, nmg_edges, mat,
            cmap=cmaps[j], norm=norm, shading="flat",
            linewidth=0.35, edgecolors="white"
        )

        # ✅ 統一 label 位置
        ax.set_xticks(mg_list)
        ax.set_yticks(nmg_list)
        ax.set_xlabel("MG ratio threshold", fontsize=12)
        ax.set_ylabel("NMG ratio threshold" if j in (0, 3) else "", fontsize=12)
        ax.set_title(lab, fontsize=14)
        _panel_label(ax, label=f"({chr(97+j)})", x=-0.12, y=1.02)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

        # ✅ 小數點三位
        mid = 0.5 * (norm.vmin + norm.vmax)
        for r_i, nmg in enumerate(nmg_list):
            for c_i, mg in enumerate(mg_list):
                val = mat[r_i, c_i]
                if np.isnan(val):
                    continue
                ax.text(
                    mg, nmg, f"{val:.3f}",
                    ha="center", va="center", fontsize=9.5,
                    color=("white" if val >= mid else "black")
                )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad=0.12)
        cb = fig.colorbar(quad, cax=cax)
        # cb.set_label("Mean share", fontsize=11)
        cb.ax.tick_params(labelsize=9, length=3.5)
        for spine in cax.spines.values():
            spine.set_linewidth(0.9)

    fig.suptitle("Action shares (mean) across MG–NMG threshold grid", fontsize=18, y=0.99)
    _ensure_dir(Path(out_png).parent)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
