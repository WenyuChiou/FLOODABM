# -*- coding: utf-8 -*-
"""
Unified Sensitivity Analysis Runner for FLOODABM
==================================================

Runs four SA experiments:
  SA1: TP Decay Rate  (tau_inf scale factors)
  SA2: Initial FI Rates  (insurance uptake multipliers)
  SA3: Flood-Prone Threshold  (reclassify tracts)
  SA4: RT × TP Decay Interaction  (2-parameter grid)

Usage:
    python run_sensitivity_analysis.py                    # run all
    python run_sensitivity_analysis.py --sa sa1           # run SA1 only
    python run_sensitivity_analysis.py --sa sa1 --reuse   # skip existing
    python run_sensitivity_analysis.py --dry-run           # preview commands
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.stdout.reconfigure(encoding="utf-8")

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
YAML_PATH = ROOT / "config" / "abm_params.yaml"
CSV_PATH = ROOT / "config" / "households_for_abm.csv"
DEPTHS_JSON = ROOT / "config" / "overall_md_mean_by_tract_2011_2023.json"
SA_OUT = ROOT / "outputs" / "experiments" / "sensitivity_analysis"

# ── SA parameter definitions ────────────────────────────────────────────

SA1_SCALES = [0.5, 0.75, 1.0, 1.5, 2.0]
SA2_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]
SA3_THRESHOLDS = [1, 7, 13]  # years of flooding → FP classification
SA4_RT_VALUES = [0.05, 0.10, 0.20]
SA4_DECAY_SCALES = [0.5, 1.0, 2.0]

# Baseline tau_inf values from YAML
BASELINE_TAU_INF_OWNER = 33.2589
BASELINE_TAU_INF_RENTER = 46.0431


# ── YAML helpers ─────────────────────────────────────────────────────────

def load_yaml() -> dict:
    """Load YAML config as dict."""
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


def save_yaml(cfg: dict) -> None:
    """Write dict back to YAML, preserving key order."""
    YAML_PATH.write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )


def backup_yaml() -> str:
    """Return original YAML text for later restore."""
    return YAML_PATH.read_text(encoding="utf-8")


def restore_yaml(original_text: str) -> None:
    """Restore YAML from saved text."""
    YAML_PATH.write_text(original_text, encoding="utf-8")


def backup_csv() -> bytes:
    """Return CSV bytes for later restore."""
    return CSV_PATH.read_bytes()


def restore_csv(original_bytes: bytes) -> None:
    """Restore CSV from saved bytes."""
    CSV_PATH.write_bytes(original_bytes)


# ── run dir helpers (from sa_ratio_threshold.py) ─────────────────────────

def _normalize_run_dir(run_dir: Path) -> Path:
    """Drill into nested baseline/ subdirectories."""
    for _ in range(5):
        b = run_dir / "baseline"
        if not b.exists():
            break
        if (b / "tp_traj.csv").exists() or (b / "finance").exists():
            run_dir = b
            continue
        if (b / "baseline").exists():
            run_dir = b
            continue
        break
    return run_dir


def _has_outputs(run_dir: Path) -> bool:
    """Check for complete outputs."""
    rd = _normalize_run_dir(run_dir)
    return (rd / "tp_traj.csv").exists() or (rd / "finance").exists()


# ── metric extraction (from run_montecarlo.py) ──────────────────────────

def collect_run_metrics(run_dir: Path) -> dict:
    """Extract key metrics from a completed run directory."""
    rd = _normalize_run_dir(run_dir)
    metrics = {}

    # --- Action shares (final year, population-weighted) ---
    dec_dir = rd / "decisions"
    if dec_dir.exists():
        # Try all-years file first
        all_f = dec_dir / "action_share_owner_renter_tract_all_years.csv"
        if all_f.exists():
            act = pd.read_csv(all_f, encoding="utf-8")
        else:
            files = sorted(dec_dir.glob("action_share_owner_renter_tract_2*.csv"))
            act = pd.concat([pd.read_csv(f, encoding="utf-8") for f in files],
                            ignore_index=True) if files else pd.DataFrame()

        if not act.empty:
            final_yr = act["year"].max()
            a = act[act["year"] == final_yr]

            # Owner shares
            if "owner_N_total" in a.columns:
                ow = a["owner_N_total"].sum()
                if ow > 0:
                    for col in ["owner_share_FI", "owner_share_EH",
                                "owner_share_BP", "owner_share_DN"]:
                        if col in a.columns:
                            metrics[col] = round(
                                (a[col] * a["owner_N_total"]).sum() / ow, 4
                            )

            # Renter shares
            if "renter_N_total" in a.columns:
                rw = a["renter_N_total"].sum()
                if rw > 0:
                    for col in ["renter_share_FI", "renter_share_RL",
                                "renter_share_DN"]:
                        if col in a.columns:
                            metrics[col] = round(
                                (a[col] * a["renter_N_total"]).sum() / rw, 4
                            )

    # --- TP trajectory (final year mean) ---
    tp_path = rd / "tp_traj.csv"
    if tp_path.exists():
        tp = pd.read_csv(tp_path, encoding="utf-8")
        tp_yr = tp.groupby("year")[["TP_owner", "TP_renter"]].mean()
        if not tp_yr.empty:
            final = tp_yr.iloc[-1]
            metrics["tp_final_owner"] = round(float(final["TP_owner"]), 4)
            metrics["tp_final_renter"] = round(float(final["TP_renter"]), 4)
            # Store full trajectory for time-series plotting
            for yr in tp_yr.index:
                metrics[f"tp_owner_{yr}"] = round(float(tp_yr.loc[yr, "TP_owner"]), 4)
                metrics[f"tp_renter_{yr}"] = round(float(tp_yr.loc[yr, "TP_renter"]), 4)

    # --- Cumulative damage ---
    dmg_path = rd / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv"
    if dmg_path.exists():
        try:
            dmg = pd.read_csv(dmg_path, dtype={"tract_geoid": str})
            if "both_usd" in dmg.columns:
                metrics["total_damage_usd"] = float(dmg["both_usd"].sum())
            if "owner_usd" in dmg.columns:
                metrics["owner_damage_usd"] = float(dmg["owner_usd"].sum())
            if "renter_usd" in dmg.columns:
                metrics["renter_damage_usd"] = float(dmg["renter_usd"].sum())
        except Exception as exc:
            print(f"  [WARN] Failed to read {f.name}: {exc}")

    # --- Finance totals ---
    fin_dir = rd / "finance"
    if fin_dir.exists():
        total_payout = 0.0
        total_premium = 0.0
        total_oop = 0.0
        for f in sorted(fin_dir.glob("finance_tract_2*.csv")):
            try:
                df = pd.read_csv(f, dtype={"tract_geoid": str})
                if "payout_total_usd" in df.columns:
                    total_payout += df["payout_total_usd"].sum()
                if "premium_total_usd" in df.columns:
                    total_premium += df["premium_total_usd"].sum()
                if "oop_total_usd" in df.columns:
                    total_oop += df["oop_total_usd"].sum()
            except Exception:
                pass
        metrics["cumulative_payout_usd"] = total_payout
        metrics["cumulative_premium_usd"] = total_premium
        metrics["cumulative_oop_usd"] = total_oop

    return metrics


# ── simulation runner ────────────────────────────────────────────────────

def run_sim(tag: str, out_dir: Path, extra_args: list[str] | None = None,
            reuse: bool = False, dry_run: bool = False) -> tuple[Path, float]:
    """Run a single FLOODABM simulation.

    Returns: (run_dir, elapsed_seconds)
    """
    if reuse and _has_outputs(out_dir):
        print(f"  [reuse] {tag} — outputs exist, skipping")
        return out_dir, 0.0

    if dry_run:
        cmd_str = " ".join(
            [sys.executable, "main.py", "--scenario", "baseline",
             "--output-mode", "summary", "--no-plots"] + (extra_args or [])
        )
        print(f"  [dry-run] {tag}: {cmd_str}")
        print(f"            out → {out_dir}")
        return out_dir, 0.0

    print(f"  [run]   {tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["FLOODABM_OUTPUT_ROOT"] = str(out_dir)
    env["FLOODABM_OUTPUT_MODE"] = "summary"

    cmd = [
        sys.executable, "main.py",
        "--scenario", "baseline",
        "--output-mode", "summary",
        "--no-plots",
    ] + (extra_args or [])

    t0 = time.time()
    ret = subprocess.call(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - t0

    if ret != 0:
        print(f"  [WARN] {tag} exited with code {ret}")
    else:
        print(f"  [OK]   {tag} in {elapsed:.1f}s")

    return out_dir, elapsed


# ── SA1: TP Decay Rate ──────────────────────────────────────────────────

def run_sa1(reuse: bool = False, dry_run: bool = False) -> list[dict]:
    """SA1: Sweep tau_inf scale factors."""
    print("\n" + "=" * 60)
    print("SA1: TP Decay Rate (tau_inf scale)")
    print(f"  Scales: {SA1_SCALES}")
    print("=" * 60)

    sa1_dir = SA_OUT / "sa1_tp_decay"
    sa1_dir.mkdir(parents=True, exist_ok=True)

    yaml_backup = backup_yaml()
    results = []

    try:
        for scale in SA1_SCALES:
            tag = f"scale_{scale:.2f}".replace(".", "")
            out_dir = sa1_dir / tag

            # Patch YAML: multiply tau_inf by scale factor
            cfg = load_yaml()
            cfg["tp_decay_params"]["owner"]["tau_inf"] = round(
                BASELINE_TAU_INF_OWNER * scale, 4
            )
            cfg["tp_decay_params"]["renter"]["tau_inf"] = round(
                BASELINE_TAU_INF_RENTER * scale, 4
            )
            save_yaml(cfg)

            run_dir, elapsed = run_sim(
                f"SA1 {tag}", out_dir, reuse=reuse, dry_run=dry_run
            )

            if not dry_run:
                m = collect_run_metrics(run_dir)
                m["sa"] = "sa1_tp_decay"
                m["parameter"] = "tau_inf_scale"
                m["value"] = scale
                m["tau_inf_owner"] = round(BASELINE_TAU_INF_OWNER * scale, 4)
                m["tau_inf_renter"] = round(BASELINE_TAU_INF_RENTER * scale, 4)
                m["elapsed_s"] = elapsed
                results.append(m)
    finally:
        restore_yaml(yaml_backup)

    return results


# ── SA2: Initial FI Rates ───────────────────────────────────────────────

def run_sa2(reuse: bool = False, dry_run: bool = False) -> list[dict]:
    """SA2: Sweep insurance uptake rate multipliers."""
    print("\n" + "=" * 60)
    print("SA2: Initial FI Rates (multiplier)")
    print(f"  Multipliers: {SA2_MULTIPLIERS}")
    print("=" * 60)

    sa2_dir = SA_OUT / "sa2_fi_rates"
    sa2_dir.mkdir(parents=True, exist_ok=True)

    yaml_backup = backup_yaml()
    csv_backup = backup_csv()
    results = []

    try:
        cfg_orig = yaml.safe_load(yaml_backup)
        orig_rates = cfg_orig["insurance_init"]["take_rate_by_tract_group"]

        for mult in SA2_MULTIPLIERS:
            tag = f"mult_{mult:.2f}".replace(".", "")
            out_dir = sa2_dir / tag

            if reuse and _has_outputs(out_dir):
                print(f"  [reuse] SA2 {tag} — outputs exist, skipping")
                if not dry_run:
                    m = collect_run_metrics(out_dir)
                    m["sa"] = "sa2_fi_rates"
                    m["parameter"] = "fi_rate_multiplier"
                    m["value"] = mult
                    m["elapsed_s"] = 0.0
                    results.append(m)
                continue

            # Patch YAML: multiply all take rates
            cfg = load_yaml()
            new_rates = {}
            for tract, grp_rates in orig_rates.items():
                new_rates[tract] = {}
                for grp, rate in grp_rates.items():
                    new_rates[tract][grp] = round(
                        min(float(rate) * mult, 1.0), 4
                    )
            cfg["insurance_init"]["take_rate_by_tract_group"] = new_rates
            save_yaml(cfg)

            # Regenerate household CSV (has_FI depends on rates)
            if not dry_run:
                print(f"  [regen] Regenerating households CSV for mult={mult}")
                restore_csv(csv_backup)  # Start from original CSV
                subprocess.check_call(
                    [sys.executable, "generate_household_psych.py",
                     "--seed", "2025"],
                    cwd=str(ROOT),
                )

            run_dir, elapsed = run_sim(
                f"SA2 {tag}", out_dir, reuse=False, dry_run=dry_run
            )

            if not dry_run:
                m = collect_run_metrics(run_dir)
                m["sa"] = "sa2_fi_rates"
                m["parameter"] = "fi_rate_multiplier"
                m["value"] = mult
                m["elapsed_s"] = elapsed
                results.append(m)
    finally:
        restore_yaml(yaml_backup)
        restore_csv(csv_backup)
        # Regenerate with original config
        if not dry_run:
            print("  [restore] Regenerating original households CSV")
            subprocess.call(
                [sys.executable, "generate_household_psych.py", "--seed", "2025"],
                cwd=str(ROOT),
            )

    return results


# ── SA3: Flood-Prone Threshold ──────────────────────────────────────────

def count_flood_years_per_tract() -> dict[str, int]:
    """Count non-zero depth years per tract from the depths JSON."""
    data = json.loads(DEPTHS_JSON.read_text(encoding="utf-8"))
    counts = {}
    for row in data:
        # Ensure tract is 11-digit string regardless of JSON type
        tract = str(int(float(row["CensusTract"]))).zfill(11)
        n = sum(1 for k, v in row.items()
                if k.endswith("_mean") and _safe_positive(v))
        counts[tract] = n
    return counts


def _safe_positive(v) -> bool:
    """Check if value is a positive number, handling None/null."""
    try:
        return float(v) > 0
    except (TypeError, ValueError):
        return False


def classify_tracts(flood_counts: dict[str, int],
                    threshold: int) -> tuple[set[str], set[str]]:
    """Classify tracts as flood-prone or not based on threshold.

    Returns: (flood_prone_tracts, non_flood_prone_tracts)
    """
    fp = {t for t, n in flood_counts.items() if n >= threshold}
    nfp = {t for t, n in flood_counts.items() if n < threshold}
    return fp, nfp


def run_sa3(reuse: bool = False, dry_run: bool = False) -> list[dict]:
    """SA3: Sweep flood-prone classification threshold."""
    print("\n" + "=" * 60)
    print("SA3: Flood-Prone Threshold (years of flooding)")
    print(f"  Thresholds: {SA3_THRESHOLDS}")
    print("=" * 60)

    sa3_dir = SA_OUT / "sa3_fp_threshold"
    sa3_dir.mkdir(parents=True, exist_ok=True)

    yaml_backup = backup_yaml()
    csv_backup = backup_csv()

    flood_counts = count_flood_years_per_tract()
    print(f"  Tract flood counts: {flood_counts}")

    # Baseline FP/NFP rates from YAML
    cfg_orig = yaml.safe_load(yaml_backup)
    # FP rates: owner=0.25, renter=0.08; NFP rates: owner=0.03, renter=0.01
    FP_OWNER, FP_RENTER = 0.25, 0.08
    NFP_OWNER, NFP_RENTER = 0.03, 0.01

    results = []
    try:
        for thr in SA3_THRESHOLDS:
            tag = f"thr_{thr:02d}"
            out_dir = sa3_dir / tag

            fp_tracts, nfp_tracts = classify_tracts(flood_counts, thr)
            print(f"\n  Threshold={thr}: {len(fp_tracts)} FP tracts, "
                  f"{len(nfp_tracts)} NFP tracts")

            if reuse and _has_outputs(out_dir):
                print(f"  [reuse] SA3 {tag} — outputs exist, skipping")
                if not dry_run:
                    m = collect_run_metrics(out_dir)
                    m["sa"] = "sa3_fp_threshold"
                    m["parameter"] = "fp_year_threshold"
                    m["value"] = thr
                    m["n_fp_tracts"] = len(fp_tracts)
                    m["n_nfp_tracts"] = len(nfp_tracts)
                    m["elapsed_s"] = 0.0
                    results.append(m)
                continue

            # Patch YAML: reclassify tracts
            cfg = load_yaml()
            new_rates = {}
            for tract in cfg["insurance_init"]["take_rate_by_tract_group"]:
                tract_str = str(tract).zfill(11)
                if tract_str in fp_tracts:
                    new_rates[tract] = {"owner": FP_OWNER, "renter": FP_RENTER}
                else:
                    new_rates[tract] = {"owner": NFP_OWNER, "renter": NFP_RENTER}
            cfg["insurance_init"]["take_rate_by_tract_group"] = new_rates
            save_yaml(cfg)

            # Regenerate household CSV
            if not dry_run:
                print(f"  [regen] Regenerating households CSV for thr={thr}")
                restore_csv(csv_backup)
                subprocess.check_call(
                    [sys.executable, "generate_household_psych.py",
                     "--seed", "2025"],
                    cwd=str(ROOT),
                )

            run_dir, elapsed = run_sim(
                f"SA3 {tag}", out_dir, reuse=False, dry_run=dry_run
            )

            if not dry_run:
                m = collect_run_metrics(run_dir)
                m["sa"] = "sa3_fp_threshold"
                m["parameter"] = "fp_year_threshold"
                m["value"] = thr
                m["n_fp_tracts"] = len(fp_tracts)
                m["n_nfp_tracts"] = len(nfp_tracts)
                m["elapsed_s"] = elapsed
                results.append(m)
    finally:
        restore_yaml(yaml_backup)
        restore_csv(csv_backup)
        if not dry_run:
            print("  [restore] Regenerating original households CSV")
            subprocess.call(
                [sys.executable, "generate_household_psych.py", "--seed", "2025"],
                cwd=str(ROOT),
            )

    return results


# ── SA4: RT × TP Decay Interaction ──────────────────────────────────────

def run_sa4(reuse: bool = False, dry_run: bool = False) -> list[dict]:
    """SA4: 3×3 grid of RT × tau_inf scale."""
    print("\n" + "=" * 60)
    print("SA4: RT × TP Decay Interaction")
    print(f"  RT values:    {SA4_RT_VALUES}")
    print(f"  Decay scales: {SA4_DECAY_SCALES}")
    print(f"  Grid: {len(SA4_RT_VALUES)}×{len(SA4_DECAY_SCALES)} = "
          f"{len(SA4_RT_VALUES) * len(SA4_DECAY_SCALES)} runs")
    print("=" * 60)

    sa4_dir = SA_OUT / "sa4_interaction"
    sa4_dir.mkdir(parents=True, exist_ok=True)

    yaml_backup = backup_yaml()
    results = []

    try:
        for rt in SA4_RT_VALUES:
            for dscale in SA4_DECAY_SCALES:
                rt_str = f"{rt:.2f}".replace(".", "")
                ds_str = f"{dscale:.1f}".replace(".", "")
                tag = f"RT{rt_str}_D{ds_str}"
                out_dir = sa4_dir / tag

                # Patch YAML: tau_inf
                cfg = load_yaml()
                cfg["tp_decay_params"]["owner"]["tau_inf"] = round(
                    BASELINE_TAU_INF_OWNER * dscale, 4
                )
                cfg["tp_decay_params"]["renter"]["tau_inf"] = round(
                    BASELINE_TAU_INF_RENTER * dscale, 4
                )
                save_yaml(cfg)

                # RT via CLI args
                extra = [
                    "--thr-owner", str(rt),
                    "--thr-renter", str(rt),
                ]

                run_dir, elapsed = run_sim(
                    f"SA4 {tag}", out_dir, extra_args=extra,
                    reuse=reuse, dry_run=dry_run,
                )

                if not dry_run:
                    m = collect_run_metrics(run_dir)
                    m["sa"] = "sa4_interaction"
                    m["parameter"] = "RT_x_decay"
                    m["rt"] = rt
                    m["decay_scale"] = dscale
                    m["value"] = f"RT={rt},D={dscale}"
                    m["elapsed_s"] = elapsed
                    results.append(m)
    finally:
        restore_yaml(yaml_backup)

    return results


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Unified Sensitivity Analysis for FLOODABM"
    )
    ap.add_argument(
        "--sa", choices=["sa1", "sa2", "sa3", "sa4", "all"],
        default="all", help="Which SA to run (default: all)"
    )
    ap.add_argument("--reuse", action="store_true",
                    help="Skip runs whose outputs already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without running")
    args = ap.parse_args()

    SA_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FLOODABM Unified Sensitivity Analysis")
    print(f"  Experiment: {args.sa}")
    print(f"  Reuse: {args.reuse}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"  Output: {SA_OUT}")
    print("=" * 60)

    t0 = time.time()
    all_results = []

    # Dispatch
    experiments = {
        "sa1": run_sa1,
        "sa2": run_sa2,
        "sa3": run_sa3,
        "sa4": run_sa4,
    }

    if args.sa == "all":
        for name, func in experiments.items():
            results = func(reuse=args.reuse, dry_run=args.dry_run)
            all_results.extend(results)
    else:
        results = experiments[args.sa](reuse=args.reuse, dry_run=args.dry_run)
        all_results.extend(results)

    # Save unified metrics
    if all_results:
        df = pd.DataFrame(all_results)
        out_csv = SA_OUT / "sa_all_metrics.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n{'='*60}")
        print(f"Saved {len(df)} rows to {out_csv}")
        print(f"Columns: {list(df.columns)}")
    elif not args.dry_run:
        print("\n[WARN] No results collected")

    total_time = time.time() - t0
    print(f"\nTotal elapsed: {total_time/60:.1f} min")
    print("Done!")


if __name__ == "__main__":
    main()
