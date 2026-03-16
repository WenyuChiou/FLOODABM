"""
NFIP Validation: compare ABM simulation payout/damage vs OpenFEMA claims.

Usage:
    python validate_nfip.py [--sim-root <path>] [--obs <path>] [--out <path>]

Defaults use the current baseline outputs.
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ── paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DEFAULT_SIM = ROOT / "outputs" / "baseline" / "baseline" / "finance"
DEFAULT_OBS = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\flood insurance\openfema_claims_summary_by_tract_year.csv")
DEFAULT_OUT = ROOT / "outputs" / "baseline" / "baseline" / "visualization" / "validation"

YEARS = range(2011, 2024)

# ── county lookup (tract prefix → county) ─────────────────────────
COUNTY_MAP = {
    "34013": "Essex",
    "34027": "Passaic",
    "34031": "Morris",
}

def _county(tract: str) -> str:
    return COUNTY_MAP.get(tract[:5], "Unknown")


# ── load simulation finance ────────────────────────────────────────
def load_sim(sim_dir: Path) -> pd.DataFrame:
    rows = []
    for y in YEARS:
        fp = sim_dir / f"finance_tract_{y}.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, dtype={"tract_geoid": str})
        df["year"] = y
        rows.append(df[["tract_geoid", "year",
                         "gross_total_usd", "payout_total_usd",
                         "oop_total_usd", "claims",
                         "owner_gross_total_usd", "renter_gross_total_usd",
                         "payout_structure_usd", "payout_contents_usd",
                         "households"]])
    sim = pd.concat(rows, ignore_index=True)
    sim.rename(columns={"tract_geoid": "tract"}, inplace=True)
    sim["county"] = sim["tract"].apply(_county)
    return sim


# ── load OpenFEMA observed ─────────────────────────────────────────
def load_obs(obs_path: Path) -> pd.DataFrame:
    obs = pd.read_csv(obs_path, dtype={"censusTract": str})
    obs.rename(columns={
        "censusTract": "tract",
        "yearOfLoss": "year",
        "totalClaims": "obs_claims",
        "totalBuildingPaid": "obs_building_paid",
        "totalContentsPaid": "obs_contents_paid",
        "totalDamage": "obs_damage",
    }, inplace=True)
    obs["obs_payout"] = obs["obs_building_paid"] + obs["obs_contents_paid"]
    obs["county"] = obs["tract"].apply(_county)
    return obs


# ── merge ──────────────────────────────────────────────────────────
def merge_sim_obs(sim: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    # Ensure all tract-year combos from sim exist (fill obs with 0 where no claims)
    merged = sim.merge(obs[["tract", "year", "obs_payout", "obs_damage", "obs_claims"]],
                       on=["tract", "year"], how="left")
    merged[["obs_payout", "obs_damage", "obs_claims"]] = \
        merged[["obs_payout", "obs_damage", "obs_claims"]].fillna(0)
    return merged


# ── metrics ────────────────────────────────────────────────────────
def _metrics(sim_arr, obs_arr, label=""):
    mask = np.isfinite(sim_arr) & np.isfinite(obs_arr)
    s, o = sim_arr[mask], obs_arr[mask]
    n = len(s)
    if n < 2:
        return {"label": label, "n": n}

    ss_res = np.sum((s - o) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(ss_res / n)
    mae = np.mean(np.abs(s - o))
    bias = (s.sum() - o.sum()) / o.sum() if o.sum() != 0 else np.nan
    ratio = s.sum() / o.sum() if o.sum() != 0 else np.nan
    corr = np.corrcoef(s, o)[0, 1] if n > 1 else np.nan

    return {
        "label": label, "n": n,
        "sim_total": s.sum(), "obs_total": o.sum(),
        "R2": r2, "Pearson_r": corr, "RMSE": rmse, "MAE": mae,
        "relative_bias": bias, "sim_obs_ratio": ratio,
    }


# ── annual aggregation ─────────────────────────────────────────────
def annual_comparison(merged: pd.DataFrame) -> pd.DataFrame:
    agg = merged.groupby("year").agg(
        sim_payout=("payout_total_usd", "sum"),
        obs_payout=("obs_payout", "sum"),
        sim_damage=("gross_total_usd", "sum"),
        obs_damage=("obs_damage", "sum"),
        sim_claims=("claims", "sum"),
        obs_claims=("obs_claims", "sum"),
        sim_hh=("households", "sum"),
    ).reset_index()
    agg["sim_payout_per_claim"] = np.where(agg["sim_claims"] > 0,
                                            agg["sim_payout"] / agg["sim_claims"], np.nan)
    agg["obs_payout_per_claim"] = np.where(agg["obs_claims"] > 0,
                                            agg["obs_payout"] / agg["obs_claims"], np.nan)
    return agg


# ── county-year aggregation ────────────────────────────────────────
def county_year_comparison(merged: pd.DataFrame) -> pd.DataFrame:
    agg = merged.groupby(["county", "year"]).agg(
        sim_payout=("payout_total_usd", "sum"),
        obs_payout=("obs_payout", "sum"),
        sim_damage=("gross_total_usd", "sum"),
        obs_damage=("obs_damage", "sum"),
        sim_claims=("claims", "sum"),
        obs_claims=("obs_claims", "sum"),
    ).reset_index()
    return agg


# ── visualization ──────────────────────────────────────────────────
def plot_validation(annual: pd.DataFrame, county_yr: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    def _usd_fmt(x, _):
        if abs(x) >= 1e6:
            return f"${x/1e6:.0f}M"
        if abs(x) >= 1e3:
            return f"${x/1e3:.0f}K"
        return f"${x:.0f}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Fig 1: Annual payout comparison (3-county total) ──
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # (a) Annual total payout
    ax = axs[0, 0]
    ax.bar(annual["year"] - 0.2, annual["obs_payout"], width=0.4,
           label="Observed (NFIP)", color="#2196F3", alpha=0.8)
    ax.bar(annual["year"] + 0.2, annual["sim_payout"], width=0.4,
           label="Simulated (ABM)", color="#FF5722", alpha=0.8)
    ax.set_ylabel("Total Payout ($)")
    ax.set_title("(a) Annual Total Insurance Payout")
    ax.yaxis.set_major_formatter(FuncFormatter(_usd_fmt))
    ax.legend(fontsize=8)
    ax.set_xlabel("Year")

    # (b) Annual total damage
    ax = axs[0, 1]
    ax.bar(annual["year"] - 0.2, annual["obs_damage"], width=0.4,
           label="Observed", color="#2196F3", alpha=0.8)
    ax.bar(annual["year"] + 0.2, annual["sim_damage"], width=0.4,
           label="Simulated", color="#FF5722", alpha=0.8)
    ax.set_ylabel("Total Damage ($)")
    ax.set_title("(b) Annual Total Flood Damage")
    ax.yaxis.set_major_formatter(FuncFormatter(_usd_fmt))
    ax.legend(fontsize=8)
    ax.set_xlabel("Year")

    # (c) Scatter: sim vs obs payout (county-year)
    ax = axs[1, 0]
    for county, grp in county_yr.groupby("county"):
        ax.scatter(grp["obs_payout"], grp["sim_payout"], label=county, s=40, alpha=0.7)
    # 1:1 line
    lim = max(county_yr["obs_payout"].max(), county_yr["sim_payout"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="1:1")
    ax.set_xlabel("Observed Payout ($)")
    ax.set_ylabel("Simulated Payout ($)")
    ax.set_title("(c) County-Year Payout: Sim vs Obs")
    ax.xaxis.set_major_formatter(FuncFormatter(_usd_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_usd_fmt))
    ax.legend(fontsize=8)

    # (d) Standardized time series (z-score)
    ax = axs[1, 1]
    ann = annual.copy()
    for col in ["sim_payout", "obs_payout"]:
        mu, sd = ann[col].mean(), ann[col].std()
        ann[f"{col}_z"] = (ann[col] - mu) / sd if sd > 0 else 0
    ax.plot(ann["year"], ann["obs_payout_z"], "o-", color="#2196F3", label="Observed (z)")
    ax.plot(ann["year"], ann["sim_payout_z"], "s-", color="#FF5722", label="Simulated (z)")
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.set_ylabel("Standardized Payout (z-score)")
    ax.set_title("(d) Standardized Annual Payout")
    ax.legend(fontsize=8)
    ax.set_xlabel("Year")

    fig.savefig(out_dir / "nfip_validation_4panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'nfip_validation_4panel.png'}")

    # ── Fig 2: County-level time series ──
    counties = sorted(county_yr["county"].unique())
    fig, axs = plt.subplots(1, len(counties), figsize=(5 * len(counties), 4),
                            constrained_layout=True, sharey=False)
    if len(counties) == 1:
        axs = [axs]
    for ax, county in zip(axs, counties):
        grp = county_yr[county_yr["county"] == county].sort_values("year")
        ax.bar(grp["year"] - 0.2, grp["obs_payout"], width=0.4,
               label="Observed", color="#2196F3", alpha=0.8)
        ax.bar(grp["year"] + 0.2, grp["sim_payout"], width=0.4,
               label="Simulated", color="#FF5722", alpha=0.8)
        ax.set_title(f"{county} County")
        ax.set_xlabel("Year")
        ax.yaxis.set_major_formatter(FuncFormatter(_usd_fmt))
        ax.legend(fontsize=7)
    axs[0].set_ylabel("Insurance Payout ($)")
    fig.savefig(out_dir / "nfip_validation_by_county.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'nfip_validation_by_county.png'}")


# ── main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NFIP validation")
    parser.add_argument("--sim-root", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--obs", type=Path, default=DEFAULT_OBS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print("=" * 60)
    print("NFIP VALIDATION: Simulation vs OpenFEMA Claims")
    print("=" * 60)
    print(f"  Sim dir : {args.sim_root}")
    print(f"  Obs file: {args.obs}")
    print(f"  Out dir : {args.out}")
    print()

    # ── load ──
    sim = load_sim(args.sim_root)
    obs = load_obs(args.obs)
    merged = merge_sim_obs(sim, obs)

    print(f"Sim records : {len(sim)} tract-years")
    print(f"Obs records : {len(obs)} tract-years (with claims)")
    print(f"Merged      : {len(merged)} tract-years")
    print()

    # ── tract-year metrics ──
    m_payout = _metrics(merged["payout_total_usd"].values,
                        merged["obs_payout"].values, "Payout (tract-year)")
    m_damage = _metrics(merged["gross_total_usd"].values,
                        merged["obs_damage"].values, "Damage (tract-year)")
    print("── Tract-Year Level Metrics ──")
    for m in [m_payout, m_damage]:
        print(f"\n  {m['label']}:")
        for k, v in m.items():
            if k in ("label", "n"):
                continue
            if isinstance(v, float):
                print(f"    {k:20s}: {v:>14.4f}")

    # ── annual ──
    annual = annual_comparison(merged)
    print("\n── Annual Aggregate ──")
    print(annual[["year", "sim_payout", "obs_payout", "sim_damage", "obs_damage",
                   "sim_claims", "obs_claims"]].to_string(index=False))

    m_ann_payout = _metrics(annual["sim_payout"].values,
                            annual["obs_payout"].values, "Payout (annual)")
    m_ann_damage = _metrics(annual["sim_damage"].values,
                            annual["obs_damage"].values, "Damage (annual)")
    print(f"\n  Annual payout — R²={m_ann_payout.get('R2', 'N/A'):.4f}, "
          f"r={m_ann_payout.get('Pearson_r', 'N/A'):.4f}, "
          f"ratio={m_ann_payout.get('sim_obs_ratio', 'N/A'):.4f}")
    print(f"  Annual damage — R²={m_ann_damage.get('R2', 'N/A'):.4f}, "
          f"r={m_ann_damage.get('Pearson_r', 'N/A'):.4f}, "
          f"ratio={m_ann_damage.get('sim_obs_ratio', 'N/A'):.4f}")

    # ── county-year ──
    county_yr = county_year_comparison(merged)

    # ── event-specific ──
    print("\n── Event-Specific Comparison ──")
    for event_year, event_name in [(2011, "Hurricane Irene"), (2021, "Hurricane Ida")]:
        yr = merged[merged["year"] == event_year]
        s_pay = yr["payout_total_usd"].sum()
        o_pay = yr["obs_payout"].sum()
        s_dmg = yr["gross_total_usd"].sum()
        o_dmg = yr["obs_damage"].sum()
        ratio_pay = s_pay / o_pay if o_pay > 0 else np.nan
        ratio_dmg = s_dmg / o_dmg if o_dmg > 0 else np.nan
        print(f"\n  {event_year} {event_name}:")
        print(f"    Sim payout: ${s_pay:>14,.0f}  |  Obs payout: ${o_pay:>14,.0f}  |  ratio: {ratio_pay:.2f}x")
        print(f"    Sim damage: ${s_dmg:>14,.0f}  |  Obs damage: ${o_dmg:>14,.0f}  |  ratio: {ratio_dmg:.2f}x")

    # ── save CSV ──
    args.out.mkdir(parents=True, exist_ok=True)
    annual.to_csv(args.out / "nfip_annual_comparison.csv", index=False, encoding="utf-8-sig")
    county_yr.to_csv(args.out / "nfip_county_year_comparison.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(args.out / "nfip_tract_year_merged.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → CSVs saved to {args.out}")

    # ── plots ──
    print("\nGenerating plots...")
    plot_validation(annual, county_yr, args.out)

    print("\n✅ NFIP validation complete.")


if __name__ == "__main__":
    main()
