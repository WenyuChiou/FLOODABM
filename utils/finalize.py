# -*- coding: utf-8 -*-
"""
Finalization and Aggregation Module
====================================

Post-simulation aggregation of outputs and visualization plot generation.
Separated from main.py for cleaner code organization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils.sa_output_manager import SAOutputManager


def finalize_and_plot(
    years: list[int],
    output_dir: Path,
    fin_dir: Path,
    dec_dir: Path,
    vis_dir: Path,
    cfg: dict,
    out_root: Path,
    args: Any,
    tp_rows: list[dict],
    flood_rows: list[dict],
    output_mgr: SAOutputManager = None,
) -> None:
    """
    Aggregate per-year outputs and generate plots.
    
    Args:
        years: List of simulation years
        output_dir: Main output directory for scenario
        fin_dir: Finance output directory
        dec_dir: Decisions output directory
        vis_dir: Visualization output directory
        cfg: Configuration dictionary
        out_root: Root output directory (parent of scenario folders)
        args: CLI arguments namespace
        tp_rows: TP trajectory data rows
        flood_rows: Flood event data rows
        output_mgr: Output manager for storage optimization
    """
    if output_mgr is None:
        output_mgr = SAOutputManager(mode='full')
    
    # === Aggregate flood damage ===
    fd_dir = output_dir / "vulnerability" / "flood_damage"
    fd_dir.mkdir(parents=True, exist_ok=True)
    
    def _read_all(tag: str) -> pd.DataFrame:
        parts = sorted(fd_dir.glob(f"flood_damage_tract_{tag}_*.csv"))
        parts = [p for p in parts if not p.stem.endswith("_ALL") and not p.stem.endswith("_years")]
        if not parts:
            return pd.DataFrame(columns=["year", "tract_geoid", "damage_usd"])
        buf = []
        for p in parts:
            try:
                df = pd.read_csv(p)
                df["year"] = pd.to_numeric(df.get("year"), errors="coerce").astype("Int64")
                df["tract_geoid"] = df["tract_geoid"].astype(str)
                df["damage_usd"] = pd.to_numeric(df["damage_usd"], errors="coerce").fillna(0.0)
                buf.append(df[["year", "tract_geoid", "damage_usd"]])
            except Exception:
                continue
        return pd.concat(buf, ignore_index=True) if buf else pd.DataFrame(columns=["year", "tract_geoid", "damage_usd"])
    
    own = _read_all("owner").rename(columns={"damage_usd": "owner_usd"})
    ren = _read_all("renter").rename(columns={"damage_usd": "renter_usd"})
    bot = _read_all("both").rename(columns={"damage_usd": "both_usd"})
    
    all_years_df = own.merge(ren, on=["year", "tract_geoid"], how="outer").merge(bot, on=["year", "tract_geoid"], how="outer")
    for c in ["owner_usd", "renter_usd", "both_usd"]:
        all_years_df[c] = pd.to_numeric(all_years_df[c], errors="coerce").fillna(0.0)
    all_years_df = all_years_df.sort_values(["year", "tract_geoid"]).reset_index(drop=True)
    
    if output_mgr.should_save('flood_damage_tract_ALL_years'):
        all_years_df.to_csv(fd_dir / "flood_damage_tract_ALL_years.csv", index=False, encoding="utf-8-sig")
        print("[ok] wrote:", fd_dir / "flood_damage_tract_ALL_years.csv")
    
    # === Aggregate finance tract results ===
    try:
        tract_all = []
        for y in years:
            p = fin_dir / f"finance_tract_{y}.csv"
            if p.exists():
                tdf = pd.read_csv(p)
                tdf.insert(1, "year", y)
                tract_all.append(tdf)
        if tract_all and output_mgr.should_save('finance_tract_all_years'):
            pd.concat(tract_all, ignore_index=True).to_csv(
                fin_dir / "finance_tract_all_years.csv", index=False, encoding="utf-8-sig"
            )
    except Exception as e:
        print("[warn] combine tract finance failed:", e)
    
    # === Aggregate action shares ===
    try:
        all_split = []
        for y in years:
            p = dec_dir / f"action_share_owner_renter_tract_{y}.csv"
            if p.exists():
                all_split.append(pd.read_csv(p))
        if all_split and output_mgr.should_save('action_share_all_years'):
            pd.concat(all_split, ignore_index=True).to_csv(
                dec_dir / "action_share_owner_renter_tract_all_years.csv", index=False, encoding="utf-8-sig"
            )
    except Exception as e:
        print("[warn] combine owner/renter split failed:", e)
    
    # === Save TP trajectory ===
    tp_traj = pd.DataFrame(tp_rows)
    if output_mgr.should_save('tp_trajectory'):
        tp_traj.to_csv(output_dir / "tp_traj.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(flood_rows).to_csv(output_dir / "flood_years_by_tract.csv", index=False, encoding="utf-8-sig")
    
    # === Generate plots ===
    from utils.plots import plot_all_outputs
    plot_all_outputs(tp_traj, fin_dir, vis_dir)
    
    # === Optional: scenario comparison figure ===
    if getattr(args, "compare_flood_or", False):
        from utils.plots_comparision_scenario import (
            plot_cum_payoutrate_and_damage_by_group,
            export_spatial_inputs_cumrate_perhh,
        )
        sev = None
        if getattr(args, "compare_severe_years", "").strip():
            sev = [int(s) for s in args.compare_severe_years.split(",") if s.strip().isdigit()]
        try:
            # out_root should be at model_dir level (e.g., outputs/baseline/) not root level
            model_level_root = output_dir.parent  # outputs/{model_dir}
            plot_cum_payoutrate_and_damage_by_group(
                out_root=model_level_root, vis_dir=vis_dir,
                severe_years=sev or [2011, 2014, 2021],
                save_name="cum_payoutrate_and_damage_by_group",
            )
        except Exception as e:
            print("[warn] cum payout-rate + damage figure failed:", e)
        try:
            # out_root should be at model_dir level (e.g., outputs/baseline/) not root level
            model_level_root = output_dir.parent  # outputs/{model_dir}
            export_spatial_inputs_cumrate_perhh(
                out_root=model_level_root,
                out_csv=vis_dir / "spatial_inputs_cumrate_perhh_early_late.csv",
                early=(2011, 2016), late=(2017, 2023), cfg=cfg, late_is_cumulative=False,
            )
        except Exception as e:
            print("[warn] spatial export failed:", e)
    
    # === Export Excel files ===
    try:
        from utils.excel_output import export_all_excel
        export_all_excel(output_dir)
        print("[OK] Excel exports saved to:", output_dir / "excel")
    except Exception as e:
        print("[warn] Excel export failed:", e)
    
    print("[OK] Runner finished with centralized plotting under 'visualization'.")
    print("[OK] All done.")
