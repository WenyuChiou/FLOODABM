# -*- coding: utf-8 -*-
"""
Monte Carlo runner for FLOODABM.
Varies the main simulation seed to quantify stochastic variability.

Usage:
    python run_montecarlo.py --n-runs 15 --scenario baseline
    python run_montecarlo.py --n-runs 15 --scenario baseline --start-seed 42
"""
import sys, os, shutil, time, argparse, re
import numpy as np, pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
YAML_PATH = ROOT / "config" / "abm_params.yaml"
MC_OUT = ROOT / "outputs" / "montecarlo_v2"


def patch_seed_in_yaml(yaml_text: str, new_seed: int) -> str:
    """Replace top-level 'seed: XXXX' in YAML text."""
    return re.sub(r'^(seed:\s*)\d+', rf'\g<1>{new_seed}', yaml_text, count=1, flags=re.MULTILINE)


def collect_run_metrics(run_dir: Path) -> dict:
    """Extract key metrics from a completed run directory."""
    metrics = {}

    # --- Action shares (per year) ---
    dec_dir = run_dir / "decisions"
    if dec_dir.exists():
        all_shares = []
        for f in sorted(dec_dir.glob("action_share_owner_renter_tract_*.csv")):
            if "all_years" in f.name:
                continue
            try:
                df = pd.read_csv(f, dtype={"tract_geoid": str})
                m = re.search(r'(\d{4})', f.name)
                if m:
                    df["year"] = int(m.group(1))
                    all_shares.append(df)
            except Exception:
                pass
        if all_shares:
            shares = pd.concat(all_shares, ignore_index=True)
            # Aggregate: mean across tracts per year × group
            for grp in ["owner", "renter"]:
                sub = shares[shares["group"] == grp] if "group" in shares.columns else shares
                for yr in sorted(sub["year"].unique()):
                    yr_sub = sub[sub["year"] == yr]
                    for action in ["FI", "EH", "BP", "RL", "DN"]:
                        col = f"share_{action}" if f"share_{action}" in yr_sub.columns else None
                        if col and col in yr_sub.columns:
                            # Weighted average by household count
                            hh_col = "owner_households" if grp == "owner" else "renter_households"
                            if hh_col in yr_sub.columns:
                                total_hh = yr_sub[hh_col].sum()
                                if total_hh > 0:
                                    w_avg = (yr_sub[col] * yr_sub[hh_col]).sum() / total_hh
                                    metrics[f"{grp}_{action}_{yr}"] = w_avg

    # --- Finance (per year) ---
    fin_dir = run_dir / "finance"
    if fin_dir.exists():
        for f in sorted(fin_dir.glob("finance_tract_*.csv")):
            if "all" in f.name.lower():
                continue
            m = re.search(r'(\d{4})', f.name)
            if not m:
                continue
            yr = int(m.group(1))
            try:
                df = pd.read_csv(f, dtype={"tract_geoid": str})
                for col_name, metric_name in [
                    ("payout_total_usd", "payout"),
                    ("oop_total_usd", "oop"),
                    ("premium_total_usd", "premium"),
                    ("owner_gross_total_usd", "owner_gross"),
                    ("renter_gross_total_usd", "renter_gross"),
                ]:
                    if col_name in df.columns:
                        metrics[f"{metric_name}_{yr}"] = df[col_name].sum()
                # Household counts
                for hh_col in ["owner_households", "renter_households", "households"]:
                    if hh_col in df.columns:
                        metrics[f"{hh_col}_{yr}"] = df[hh_col].sum()
            except Exception:
                pass

    # --- Damage ---
    dmg_path = run_dir / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv"
    if dmg_path.exists():
        try:
            dmg = pd.read_csv(dmg_path, dtype={"tract_geoid": str})
            for yr in sorted(dmg["year"].unique()):
                yr_sub = dmg[dmg["year"] == yr]
                metrics[f"owner_dmg_{yr}"] = yr_sub["owner_usd"].sum() if "owner_usd" in yr_sub.columns else 0
                metrics[f"renter_dmg_{yr}"] = yr_sub["renter_usd"].sum() if "renter_usd" in yr_sub.columns else 0
                metrics[f"total_dmg_{yr}"] = yr_sub["both_usd"].sum() if "both_usd" in yr_sub.columns else 0
        except Exception:
            pass

    return metrics


def main():
    ap = argparse.ArgumentParser(description="Monte Carlo runner for FLOODABM")
    ap.add_argument("--n-runs", type=int, default=15, help="Number of MC runs")
    ap.add_argument("--start-seed", type=int, default=42, help="Starting seed")
    ap.add_argument("--scenario", choices=["baseline", "worst"], default="baseline")
    ap.add_argument("--output-mode", choices=["full", "summary", "minimal"], default="summary")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = ap.parse_args()

    MC_OUT.mkdir(parents=True, exist_ok=True)

    # Read original YAML
    yaml_orig = YAML_PATH.read_text(encoding="utf-8")
    original_seed = int(re.search(r'^seed:\s*(\d+)', yaml_orig, re.MULTILINE).group(1))

    # Generate seeds
    rng = np.random.RandomState(args.start_seed)
    seeds = sorted(set(rng.randint(1000, 99999, size=args.n_runs * 2).tolist()))[:args.n_runs]

    print(f"Monte Carlo: {args.n_runs} runs, scenario={args.scenario}, seeds={seeds}")
    print(f"Output mode: {args.output_mode}")
    print(f"Output dir: {MC_OUT}")
    print("=" * 60)

    # Save seed list
    seed_df = pd.DataFrame({"run_id": range(1, len(seeds) + 1), "seed": seeds})
    seed_df.to_csv(MC_OUT / "mc_seeds.csv", index=False)

    all_metrics = []
    t0_total = time.time()

    for i, seed in enumerate(seeds):
        run_id = i + 1
        run_dir = MC_OUT / f"run_{run_id:02d}"
        print(f"\n--- Run {run_id}/{len(seeds)} (seed={seed}) ---")

        if args.dry_run:
            print(f"  [dry-run] Would run with seed={seed}, output to {run_dir}")
            continue

        # Patch YAML seed
        yaml_patched = patch_seed_in_yaml(yaml_orig, seed)
        YAML_PATH.write_text(yaml_patched, encoding="utf-8")

        # Run simulation
        t0 = time.time()
        cmd = (
            f'python main.py --scenario {args.scenario} '
            f'--output-mode {args.output_mode} --no-plots '
            f'--out-root "{run_dir}"'
        )
        print(f"  CMD: {cmd}")
        ret = os.system(cmd)
        elapsed = time.time() - t0
        print(f"  Finished in {elapsed:.1f}s (exit={ret})")

        if ret != 0:
            print(f"  [WARN] Run {run_id} failed with exit code {ret}")
            continue

        # Collect metrics
        scenario_dir = run_dir / "baseline" / args.scenario
        if not scenario_dir.exists():
            # Try alternate path
            for p in run_dir.rglob("finance"):
                scenario_dir = p.parent
                break

        metrics = collect_run_metrics(scenario_dir)
        metrics["run_id"] = run_id
        metrics["seed"] = seed
        metrics["elapsed_s"] = elapsed
        all_metrics.append(metrics)

        # Progress save
        pd.DataFrame(all_metrics).to_csv(MC_OUT / "mc_metrics_progress.csv", index=False, encoding="utf-8-sig")
        print(f"  Collected {len(metrics)} metrics")

    # Restore original YAML
    YAML_PATH.write_text(yaml_orig.replace(
        patch_seed_in_yaml(yaml_orig, seeds[-1] if seeds else original_seed),
        yaml_orig
    ), encoding="utf-8")
    # Simpler restore
    yaml_restore = patch_seed_in_yaml(
        YAML_PATH.read_text(encoding="utf-8"),
        original_seed
    )
    YAML_PATH.write_text(yaml_restore, encoding="utf-8")

    if args.dry_run:
        print("\n[dry-run] No runs executed.")
        return

    # Final aggregation
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df.to_csv(MC_OUT / "mc_metrics_all.csv", index=False, encoding="utf-8-sig")

        # Generate summary statistics
        numeric_cols = [c for c in df.columns if c not in ["run_id", "seed", "elapsed_s"]]
        summary = df[numeric_cols].describe().T
        summary.to_csv(MC_OUT / "mc_summary_stats.csv", encoding="utf-8-sig")

        total_time = time.time() - t0_total
        print(f"\n{'='*60}")
        print(f"Monte Carlo complete: {len(all_metrics)}/{len(seeds)} successful runs")
        print(f"Total time: {total_time/60:.1f} min ({total_time/len(all_metrics):.1f}s per run)")
        print(f"Results: {MC_OUT / 'mc_metrics_all.csv'}")
        print(f"Summary: {MC_OUT / 'mc_summary_stats.csv'}")
    else:
        print("[ERROR] No successful runs")


if __name__ == "__main__":
    main()
