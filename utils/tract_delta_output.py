# -*- coding: utf-8 -*-
"""
Tract-Level Action Rate Delta CSV Output
=========================================
Generate per-tract detailed CSV with action rates and deltas after flood events.
Output format matches: per_tract_detailed_all_EHadjREC_BPprev_with_prone.csv

Columns:
- tract_geoid, prone_group, tenure, action, baseline_year, year, k, first, denom, rate, baseline_rate, delta
"""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_tract_delta_csv(
    decisions_dir: Path,
    output_path: Path,
    flood_events: list[tuple[int, int]] = [(2013, 2014), (2020, 2021)],
    actions: list[str] = ["FI", "BP", "EH", "RL"],
    max_k: int = 3,
) -> pd.DataFrame:
    """Generate tract-level action rate delta CSV.
    
    Args:
        decisions_dir: Directory containing action_share_owner_renter_tract_YYYY.csv files
        output_path: Path to save the output CSV
        flood_events: List of (baseline_year, event_year) tuples
        actions: List of action codes to analyze
        max_k: Maximum k (years after baseline) to include
        
    Returns:
        DataFrame with all the delta data
    """
    # Load all years data
    all_years_file = decisions_dir / "action_share_owner_renter_tract_all_years.csv"
    if not all_years_file.exists():
        print(f"File not found: {all_years_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(all_years_file, dtype={"tract_geoid": str})
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    
    # Check for flood-prone column
    prone_col = None
    for col in ['prone_group', 'flood_prone', 'is_flood_prone']:
        if col in df.columns:
            prone_col = col
            break
    
    # If no prone column exists, derive from tract depth data or mark as unknown
    if prone_col is None:
        # Default: all tracts as 'all'
        df['prone_group'] = 'all'
        prone_col = 'prone_group'
    
    results = []
    
    for baseline_year, event_year in flood_events:
        for tenure in ['owner', 'renter']:
            for action in actions:
                rate_col = f'{tenure}_share_{action}'
                count_col = f'{tenure}_count_{action}'
                denom_col = f'{tenure}_count'
                
                if rate_col not in df.columns:
                    continue
                
                # Get baseline data
                baseline_df = df[df['year'] == baseline_year].copy()
                if baseline_df.empty:
                    continue
                
                # For each k (years after baseline)
                for k in range(max_k + 1):
                    year = baseline_year + k
                    year_df = df[df['year'] == year].copy()
                    
                    if year_df.empty:
                        continue
                    
                    # Group by tract and prone_group
                    for prone_group in df[prone_col].unique():
                        baseline_prone = baseline_df[baseline_df[prone_col] == prone_group]
                        year_prone = year_df[year_df[prone_col] == prone_group]
                        
                        if baseline_prone.empty or year_prone.empty:
                            continue
                        
                        # For each tract
                        for tract in baseline_prone['tract_geoid'].unique():
                            base_row = baseline_prone[baseline_prone['tract_geoid'] == tract]
                            year_row = year_prone[year_prone['tract_geoid'] == tract]
                            
                            if base_row.empty or year_row.empty:
                                continue
                            
                            base_rate = base_row[rate_col].values[0]
                            year_rate = year_row[rate_col].values[0]
                            
                            # Get counts if available
                            first = year_row[count_col].values[0] if count_col in year_row.columns else np.nan
                            denom = year_row[denom_col].values[0] if denom_col in year_row.columns else np.nan
                            
                            results.append({
                                'tract_geoid': tract,
                                'prone_group': prone_group,
                                'tenure': tenure,
                                'action': action,
                                'baseline_year': baseline_year,
                                'year': year,
                                'k': k,
                                'first': first,
                                'denom': denom,
                                'rate': year_rate,
                                'baseline_rate': base_rate,
                                'delta': year_rate - base_rate,
                            })
    
    result_df = pd.DataFrame(results)
    
    if not result_df.empty:
        # Sort for readability
        result_df = result_df.sort_values(
            ['tract_geoid', 'prone_group', 'tenure', 'action', 'baseline_year', 'k']
        ).reset_index(drop=True)
        
        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path} ({len(result_df)} rows)")
    else:
        print("No data generated")
    
    return result_df


def main():
    decisions_dir = Path('outputs/baseline/baseline/decisions')
    output_path = Path('outputs/baseline/baseline/per_tract_detailed_with_prone.csv')
    
    df = generate_tract_delta_csv(decisions_dir, output_path)
    
    if not df.empty:
        print("\nSample output:")
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
