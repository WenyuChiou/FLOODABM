# -*- coding: utf-8 -*-
"""
Extend Monte Carlo runs from 15 to 30.
Runs 15 additional simulations with new seeds, then merges results.
"""
import sys, os, re, time
import numpy as np, pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
YAML_PATH = ROOT / "config" / "abm_params.yaml"
MC_OUT = ROOT / "outputs" / "montecarlo_v2"

# Existing seeds
existing = pd.read_csv(MC_OUT / "mc_seeds.csv")
existing_seeds = set(existing["seed"].tolist())
print(f"Existing runs: {len(existing)}, seeds: {sorted(existing_seeds)}")

# Generate 15 new non-overlapping seeds
rng = np.random.RandomState(999)
candidates = sorted(set(rng.randint(1000, 99999, size=200).tolist()))
new_seeds = [s for s in candidates if s not in existing_seeds][:15]
print(f"New seeds ({len(new_seeds)}): {new_seeds}")

# Read original YAML
yaml_orig = YAML_PATH.read_text(encoding="utf-8")
original_seed = int(re.search(r'^seed:\s*(\d+)', yaml_orig, re.MULTILINE).group(1))

def patch_seed(text, new_seed):
    return re.sub(r'^(seed:\s*)\d+', rf'\g<1>{new_seed}', text, count=1, flags=re.MULTILINE)

def collect_run_metrics(run_dir):
    """Same as run_montecarlo.py collect_run_metrics."""
    metrics = {}
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
            for grp in ["owner", "renter"]:
                sub = shares[shares["group"] == grp] if "group" in shares.columns else shares
                for yr in sorted(sub["year"].unique()):
                    yr_sub = sub[sub["year"] == yr]
                    for action in ["FI", "EH", "BP", "RL", "DN"]:
                        col = f"share_{action}" if f"share_{action}" in yr_sub.columns else None
                        if col and col in yr_sub.columns:
                            hh_col = "owner_households" if grp == "owner" else "renter_households"
                            if hh_col in yr_sub.columns:
                                total_hh = yr_sub[hh_col].sum()
                                if total_hh > 0:
                                    w_avg = (yr_sub[col] * yr_sub[hh_col]).sum() / total_hh
                                    metrics[f"{grp}_{action}_{yr}"] = w_avg

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
                    ("payout_total_usd", "payout"), ("oop_total_usd", "oop"),
                    ("premium_total_usd", "premium"),
                    ("owner_gross_total_usd", "owner_gross"),
                    ("renter_gross_total_usd", "renter_gross"),
                ]:
                    if col_name in df.columns:
                        metrics[f"{metric_name}_{yr}"] = df[col_name].sum()
                for hh_col in ["owner_households", "renter_households", "households"]:
                    if hh_col in df.columns:
                        metrics[f"{hh_col}_{yr}"] = df[hh_col].sum()
            except Exception:
                pass

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


# Run new simulations
all_new_metrics = []
t0_total = time.time()

for i, seed in enumerate(new_seeds):
    run_id = len(existing) + i + 1  # 16, 17, ..., 30
    run_dir = MC_OUT / f"run_{run_id:02d}"
    print(f"\n{'='*50}")
    print(f"Run {run_id}/30 (seed={seed})")
    print(f"{'='*50}")

    # Patch YAML
    yaml_patched = patch_seed(yaml_orig, seed)
    YAML_PATH.write_text(yaml_patched, encoding="utf-8")

    # Run simulation
    t0 = time.time()
    cmd = f'python main.py --scenario baseline --output-mode summary --no-plots --out-root "{run_dir}"'
    print(f"  CMD: {cmd}")
    ret = os.system(cmd)
    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.1f}s (exit={ret})")

    if ret != 0:
        print(f"  [WARN] Run {run_id} failed!")
        continue

    # Collect metrics
    scenario_dir = run_dir / "baseline" / "baseline"
    if not scenario_dir.exists():
        for p in run_dir.rglob("finance"):
            scenario_dir = p.parent
            break

    metrics = collect_run_metrics(scenario_dir)
    metrics["run_id"] = run_id
    metrics["seed"] = seed
    metrics["elapsed_s"] = elapsed
    all_new_metrics.append(metrics)
    print(f"  Collected {len(metrics)} metrics ({elapsed:.1f}s)")

# Restore original YAML
YAML_PATH.write_text(patch_seed(YAML_PATH.read_text(encoding="utf-8"), original_seed), encoding="utf-8")

# Merge with existing
if all_new_metrics:
    existing_df = pd.read_csv(MC_OUT / "mc_metrics_all.csv")
    new_df = pd.DataFrame(all_new_metrics)
    merged = pd.concat([existing_df, new_df], ignore_index=True)

    # Backup old file
    (MC_OUT / "mc_metrics_all_15runs_backup.csv").write_bytes(
        (MC_OUT / "mc_metrics_all.csv").read_bytes()
    )

    # Save merged
    merged.to_csv(MC_OUT / "mc_metrics_all.csv", index=False, encoding="utf-8-sig")

    # Update seeds file
    all_seeds = list(existing["seed"]) + new_seeds[:len(all_new_metrics)]
    seed_df = pd.DataFrame({
        "run_id": range(1, len(all_seeds) + 1),
        "seed": all_seeds
    })
    seed_df.to_csv(MC_OUT / "mc_seeds.csv", index=False)

    total_time = time.time() - t0_total
    print(f"\n{'='*60}")
    print(f"Extension complete: {len(all_new_metrics)} new + {len(existing)} existing = {len(merged)} total")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Merged results: {MC_OUT / 'mc_metrics_all.csv'}")
else:
    print("[ERROR] No new runs completed successfully")
