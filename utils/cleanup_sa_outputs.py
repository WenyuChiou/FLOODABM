#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SA Output Cleanup Script
========================

Removes unnecessary files from SA experiment runs to save disk space
while preserving all data needed for plotting.

Usage:
    python cleanup_sa_outputs.py outputs/experiments

This will:
- Keep: finance_tract_all_years.csv, flood_damage_tract_ALL_years.csv, 
        action_share key years (2011,2014,2021,2023), output_summary.json
- Remove: finance_households_*.csv, states/state_*.csv, excel/*.xlsx, 
          action_share non-key years, visualization/*.png

Savings: ~99.98% disk space reduction per scenario (325MB → 57KB)
"""

from pathlib import Path
import shutil
import argparse


def get_disk_usage(directory: Path) -> float:
    """Get total disk usage in MB."""
    total = 0
    try:
        for item in directory.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    except Exception as e:
        print(f"Warning: Could not calculate size for {directory}: {e}")
    return total / (1024 * 1024)  # Convert to MB


def find_data_dir(scenario_dir: Path) -> Path:
    """Find the directory containing simulation data, handling nesting."""
    # Priority 1: Check baseline/baseline
    d = scenario_dir / "baseline" / "baseline"
    if (d / "finance").exists(): return d
    
    # Priority 2: Check baseline
    d = scenario_dir / "baseline"
    if (d / "finance").exists(): return d
    
    # Priority 3: scenario_dir itself
    if (scenario_dir / "finance").exists(): return scenario_dir
    
    # Fallback: Deep search
    for p in scenario_dir.rglob("finance"):
        if p.is_dir():
            return p.parent
            
    return None


def cleanup_scenario(scenario_dir: Path, dry_run: bool = False, keep_viz: bool = False) -> dict:
    """
    Clean up a single scenario directory.
    """
    stats = {
        'files_removed': 0,
        'dirs_removed': 0,
        'bytes_freed': 0,
        'errors': []
    }
    
    if not scenario_dir.exists():
        stats['errors'].append(f"Scenario directory not found: {scenario_dir}")
        return stats
    
    # Auto-detect data directory
    base_dir = find_data_dir(scenario_dir)
    if not base_dir:
        # Check if it's already cleaned
        if (scenario_dir / "output_summary.json").exists():
            return stats # Already cleaned or minimal
        stats['errors'].append(f"No data directory found in: {scenario_dir}")
        return stats
    
    # Define what to remove (large files not needed for SA plots)
    REMOVE_PATTERNS = [
        'finance/finance_households_*.csv',
        'decisions/decisions_mgmix_*.csv', # Household level decisions
        'states/state_*.csv',
        'states/state_next_*.csv',
        'excel/*',
        *(['visualization/*.png'] if not keep_viz else []),
        'decisions/action_share_owner_renter_tract_20*.csv', # Aggressively remove all per-year tracts
    ]
    
    # But keep specific years or the aggregate count
    KEEP_PATTERNS = [
        'decisions/action_share_owner_renter_tract_all_years.csv',
        'decisions/action_share_owner_renter_tract_2011.csv',
        'decisions/action_share_owner_renter_tract_2014.csv',
        'decisions/action_share_owner_renter_tract_2021.csv',
        'decisions/action_share_owner_renter_tract_2023.csv',
        'finance/finance_tract_all_years.csv',
    ]
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Cleaning: {scenario_dir.relative_to(scenario_dir.parent.parent)}")
    
    # Remove files matching patterns
    for pattern in REMOVE_PATTERNS:
        for file_path in base_dir.glob(pattern):
            if file_path.is_file():
                # Is it in KEEP list?
                rel = str(file_path.relative_to(base_dir)).replace('\\', '/')
                if any(rel == k or rel.endswith(k) for k in KEEP_PATTERNS):
                    continue
                
                size = file_path.stat().st_size
                stats['files_removed'] += 1
                stats['bytes_freed'] += size
                
                if dry_run:
                    # Print only very large files to avoid log flooding
                    if size > 1024*1024:
                        print(f"  Would remove: {rel} ({size/(1024*1024):.2f} MB)")
                else:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        stats['errors'].append(f"Could not remove {file_path}: {e}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Clean up SA experiment outputs to save disk space"
    )
    parser.add_argument(
        "experiments_dir",
        type=Path,
        help="Path to experiments directory (e.g., outputs/experiments)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Clean only specific scenario (e.g., SA_RT_MG0_5_NMG0_7)"
    )
    parser.add_argument(
        "--keep-viz",
        action="store_true",
        help="Keep visualization plots (timeseries, etc.). Adds ~20MB per scenario."
    )
    
    args = parser.parse_args()
    
    exp_dir = args.experiments_dir.resolve()
    if not exp_dir.exists():
        print(f"Error: Experiments directory not found: {exp_dir}")
        return 1
    
    print(f"SA Output Cleanup")
    print(f"=" * 60)
    print(f"Experiments directory: {exp_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'CLEANUP'}")
    print(f"=" * 60)
    
    # Get initial size
    initial_size = get_disk_usage(exp_dir)
    print(f"\nInitial disk usage: {initial_size:.2f} MB")
    
    # Find all SA scenario directories recursively
    if args.scenario:
        scenarios = list(exp_dir.rglob(args.scenario))
    else:
        # Look for any directory starting with SA_ anywhere under exp_dir
        scenarios = [d for d in exp_dir.rglob('SA_*') if d.is_dir()]
    
    print(f"Found {len(scenarios)} SA scenario(s)")
    
    # Clean each scenario
    total_stats = {
        'files_removed': 0,
        'dirs_removed': 0,
        'bytes_freed': 0,
        'scenarios_cleaned': 0,
        'errors': []
    }
    
    for scenario_dir in sorted(scenarios):
        stats = cleanup_scenario(scenario_dir, dry_run=args.dry_run, keep_viz=args.keep_viz)
        total_stats['files_removed'] += stats['files_removed']
        total_stats['dirs_removed'] += stats['dirs_removed']
        total_stats['bytes_freed'] += stats['bytes_freed']
        total_stats['scenarios_cleaned'] += 1
        total_stats['errors'].extend(stats['errors'])
    
    # Get final size
    final_size = get_disk_usage(exp_dir)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Cleanup Summary")
    print(f"{'=' * 60}")
    print(f"Scenarios cleaned: {total_stats['scenarios_cleaned']}")
    print(f"Files removed: {total_stats['files_removed']}")
    print(f"Directories removed: {total_stats['dirs_removed']}")
    print(f"Space freed: {total_stats['bytes_freed']/(1024*1024):.2f} MB")
    print(f"\nDisk usage before: {initial_size:.2f} MB")
    print(f"Disk usage after:  {final_size:.2f} MB")
    
    if initial_size > 0:
        reduction = ((initial_size - final_size) / initial_size) * 100
        print(f"Reduction: {reduction:.1f}%")
    
    if total_stats['errors']:
        print(f"\nErrors encountered: {len(total_stats['errors'])}")
        for error in total_stats['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(total_stats['errors']) > 5:
            print(f"  ... and {len(total_stats['errors']) - 5} more")
    
    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("DRY RUN - No files were actually deleted.")
        print("Run without --dry-run to perform cleanup.")
        print(f"{'=' * 60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
