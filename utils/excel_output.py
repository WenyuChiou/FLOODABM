# -*- coding: utf-8 -*-
"""
Excel Output Utility for FLOODABM
=================================
Output comprehensive Excel files with:
- TP (Threat Perception) changes by owner/renter and tract type
- Action rates by tenure and tract type
- Finance summary
"""
import numpy as np
import pandas as pd
from pathlib import Path


def export_tp_excel(
    tp_traj_path: Path,
    output_dir: Path,
    flood_prone_tracts: list[str] = None,
) -> None:
    """Export TP trajectory data to Excel with owner/renter and tract type labels.

    Args:
        tp_traj_path: Path to tp_traj.csv
        output_dir: Directory for Excel output
        flood_prone_tracts: List of flood-prone tract GEOIDs (optional)
    """
    if not tp_traj_path.exists():
        print(f"TP file not found: {tp_traj_path}")
        return
    
    df = pd.read_csv(tp_traj_path, dtype={"tract_geoid": str})
    
    # Add tract type (flood-prone vs non-prone)
    if flood_prone_tracts:
        df['tract_type'] = df['tract_geoid'].apply(
            lambda x: 'flood-prone' if x in flood_prone_tracts else 'non-prone'
        )
    else:
        # Default: assume all are flood-prone if no list provided
        df['tract_type'] = 'all'
    
    # Create summary by year, tract_type, and group (owner/renter)
    summary_rows = []
    for year in sorted(df['year'].unique()):
        for tract_type in df['tract_type'].unique():
            year_tract = df[(df['year'] == year) & (df['tract_type'] == tract_type)]

            for group, tp_col in [('owner', 'TP_owner'), ('renter', 'TP_renter')]:
                if tp_col not in year_tract.columns:
                    continue
                
                tp_vals = year_tract[tp_col].dropna()
                if len(tp_vals) == 0:
                    continue
                
                summary_rows.append({
                    'year': year,
                    'tract_type': tract_type,
                    'group': group,
                    'n_tracts': len(tp_vals),
                    'tp_mean': tp_vals.mean(),
                    'tp_median': tp_vals.median(),
                    'tp_std': tp_vals.std(),
                    'tp_min': tp_vals.min(),
                    'tp_max': tp_vals.max(),
                    'pct_at_max': (tp_vals >= 0.99).mean() * 100,
                })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Create delta analysis for flood events
    delta_rows = []
    flood_events = [(2013, 2014), (2020, 2021)]
    
    for pre_y, post_y in flood_events:
        for tract_type in df['tract_type'].unique():
            for group, tp_col in [('owner', 'TP_owner'), ('renter', 'TP_renter')]:
                if tp_col not in df.columns:
                    continue

                pre = df[(df['year'] == pre_y) & (df['tract_type'] == tract_type)][tp_col].dropna()
                post = df[(df['year'] == post_y) & (df['tract_type'] == tract_type)][tp_col].dropna()
                
                if len(pre) == 0 or len(post) == 0:
                    continue
                
                delta_rows.append({
                    'event_year': post_y,
                    'baseline_year': pre_y,
                    'tract_type': tract_type,
                    'group': group,
                    'pre_mean': pre.mean(),
                    'post_mean': post.mean(),
                    'delta_mean': post.mean() - pre.mean(),
                    'delta_pct': (post.mean() - pre.mean()) / pre.mean() * 100 if pre.mean() > 0 else 0,
                })
    
    delta_df = pd.DataFrame(delta_rows)
    
    # Export to Excel
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / 'tp_analysis.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Raw data
        df.to_excel(writer, sheet_name='TP_Raw', index=False)
        # Summary by year
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name='TP_Summary', index=False)
        # Delta analysis
        if not delta_df.empty:
            delta_df.to_excel(writer, sheet_name='TP_Delta', index=False)
    
    print(f"Saved: {excel_path}")


def export_action_excel(
    decisions_dir: Path,
    output_dir: Path,
    flood_prone_tracts: list[str] = None,
) -> None:
    """Export action rates to Excel with tenure and tract type labels."""
    
    all_years_file = decisions_dir / "action_share_owner_renter_tract_all_years.csv"
    if not all_years_file.exists():
        print(f"File not found: {all_years_file}")
        return
    
    df = pd.read_csv(all_years_file, dtype={"tract_geoid": str})
    
    # Add tract type
    if flood_prone_tracts:
        df['tract_type'] = df['tract_geoid'].apply(
            lambda x: 'flood-prone' if x in flood_prone_tracts else 'non-prone'
        )
    else:
        df['tract_type'] = 'all'
    
    # Summary by year, tract_type, tenure, action
    actions = ['FI', 'BP', 'EH', 'RL', 'DN']
    summary_rows = []
    
    for year in sorted(df['year'].unique()):
        for tract_type in df['tract_type'].unique():
            year_tract = df[(df['year'] == year) & (df['tract_type'] == tract_type)]
            
            for tenure in ['owner', 'renter']:
                for action in actions:
                    col = f'{tenure}_share_{action}'
                    if col not in year_tract.columns:
                        continue
                    
                    vals = year_tract[col].dropna()
                    if len(vals) == 0:
                        continue
                    
                    summary_rows.append({
                        'year': year,
                        'tract_type': tract_type,
                        'tenure': tenure,
                        'action': action,
                        'n_tracts': len(vals),
                        'rate_mean': vals.mean(),
                        'rate_median': vals.median(),
                        'rate_std': vals.std(),
                    })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Export to Excel
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / 'action_analysis.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Action_Raw', index=False)
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name='Action_Summary', index=False)
    
    print(f"Saved: {excel_path}")


def export_action_rate_delta(
    decisions_dir: Path,
    output_dir: Path,
    flood_prone_tracts: list[str] = None,
) -> None:
    """Export action rate delta analysis for all actions with baseline comparison.
    
    Outputs Excel with columns matching user requirement:
    - tract_geoid, prone_group, tenure, action, baseline_year, year, k
    - first (action count), denom (total count), rate, baseline_rate, delta
    
    Args:
        decisions_dir: Directory containing decisions CSV files
        output_dir: Directory for Excel output
        flood_prone_tracts: List of flood-prone tract GEOIDs
    """
    all_years_file = decisions_dir / "action_share_owner_renter_tract_all_years.csv"
    if not all_years_file.exists():
        print(f"File not found: {all_years_file}")
        return
    
    df = pd.read_csv(all_years_file, dtype={"tract_geoid": str})
    
    # Add prone group
    if flood_prone_tracts:
        df['prone_group'] = df['tract_geoid'].apply(
            lambda x: 'flood-prone' if x in flood_prone_tracts else 'non-prone'
        )
    else:
        df['prone_group'] = 'all'
    
    # Flood events: (baseline_year, flood_year)
    flood_events = [(2013, 2014), (2020, 2021)]
    
    # Actions to analyze
    actions = ['FI', 'BP', 'EH', 'RL', 'DN']
    tenures = ['owner', 'renter']
    
    all_rows = []
    
    for baseline_year, flood_year in flood_events:
        # Get baseline data
        base_df = df[df['year'] == baseline_year].copy()
        if base_df.empty:
            continue
        
        # For k = 0, 1, 2, 3 years after flood
        for k in range(4):
            post_year = flood_year + k
            post_df = df[df['year'] == post_year].copy()
            if post_df.empty:
                continue
            
            for tenure in tenures:
                for action in actions:
                    share_col = f'{tenure}_share_{action}'
                    count_col = f'{tenure}_count'  # Total count column if exists
                    act_count_col = f'{tenure}_count_{action}'  # Action count if exists
                    
                    if share_col not in df.columns:
                        continue
                    
                    # Process each tract
                    for tract in post_df['tract_geoid'].unique():
                        prone_group = 'flood-prone' if (flood_prone_tracts and tract in flood_prone_tracts) else 'non-prone'
                        
                        # Get baseline rate for this tract
                        base_tract = base_df[base_df['tract_geoid'] == tract]
                        if base_tract.empty:
                            continue
                        baseline_rate = base_tract[share_col].values[0] if share_col in base_tract.columns else np.nan
                        
                        # Get post rate for this tract
                        post_tract = post_df[post_df['tract_geoid'] == tract]
                        if post_tract.empty:
                            continue
                        post_rate = post_tract[share_col].values[0] if share_col in post_tract.columns else np.nan
                        
                        # Get counts if available
                        denom = None
                        first = None
                        if count_col in post_tract.columns:
                            denom = post_tract[count_col].values[0]
                        if act_count_col in post_tract.columns:
                            first = post_tract[act_count_col].values[0]
                        
                        # Calculate delta
                        if pd.notna(post_rate) and pd.notna(baseline_rate):
                            delta = post_rate - baseline_rate
                        else:
                            delta = np.nan
                        
                        all_rows.append({
                            'tract_geoid': tract,
                            'prone_group': prone_group,
                            'tenure': tenure,
                            'action': action,
                            'baseline_year': baseline_year,
                            'year': post_year,
                            'k': k,
                            'first': first,
                            'denom': denom,
                            'rate': post_rate,
                            'baseline_rate': baseline_rate,
                            'delta': delta,
                        })
    
    if not all_rows:
        print("No data for action rate delta analysis")
        return
    
    result_df = pd.DataFrame(all_rows)
    
    # Create summary aggregated by prone_group, tenure, action, baseline_year, k
    summary_rows = []
    for (prone, tenure, action, base_yr, k), grp in result_df.groupby(
        ['prone_group', 'tenure', 'action', 'baseline_year', 'k']
    ):
        rates = grp['rate'].dropna()
        deltas = grp['delta'].dropna()
        baseline_rates = grp['baseline_rate'].dropna()
        
        summary_rows.append({
            'prone_group': prone,
            'tenure': tenure,
            'action': action,
            'baseline_year': base_yr,
            'k': k,
            'year': int(base_yr) + 1 + int(k),  # flood_year + k
            'n_tracts': len(rates),
            'rate_mean': rates.mean() if len(rates) > 0 else np.nan,
            'rate_median': rates.median() if len(rates) > 0 else np.nan,
            'baseline_rate_mean': baseline_rates.mean() if len(baseline_rates) > 0 else np.nan,
            'delta_mean': deltas.mean() if len(deltas) > 0 else np.nan,
            'delta_median': deltas.median() if len(deltas) > 0 else np.nan,
            'delta_std': deltas.std() if len(deltas) > 0 else np.nan,
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Export to Excel
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / 'action_rate_delta.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='Action_Rate_All', index=False)
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"Saved: {excel_path}")


def export_fi_takeup_change(
    states_dir: Path,
    output_dir: Path,
    flood_prone_tracts: list[str] = None,
) -> None:
    """Export FI takeup change analysis for post-flood periods.
    
    Outputs Excel with:
    - tract_id, tenure, tract_type
    - FI rate changes for k=0, k=1, k=2 after each flood event
    
    Args:
        states_dir: Directory containing state_for_vuln_YYYY.csv files
        output_dir: Directory for Excel output
        flood_prone_tracts: List of flood-prone tract GEOIDs
    """
    import numpy as np
    
    if not states_dir.exists():
        print(f"States dir not found: {states_dir}")
        return
    
    # Flood events: (baseline_year, flood_year)
    flood_events = [(2013, 2014), (2020, 2021)]
    
    # Auto-detect flood-prone tracts if not provided
    if flood_prone_tracts is None:
        init_file = states_dir / "state_for_vuln_2011.csv"
        if init_file.exists():
            init_df = pd.read_csv(init_file, dtype={"tract_geoid": str})
            init_df['tract'] = init_df['tract_geoid'].astype(str)
            tract_fi = init_df.groupby('tract')['has_FI'].mean()
            flood_prone_tracts = tract_fi[tract_fi >= 0.2].index.tolist()
        else:
            flood_prone_tracts = []
    
    all_rows = []
    
    for baseline_year, flood_year in flood_events:
        # Load baseline state
        baseline_file = states_dir / f"state_for_vuln_{baseline_year}.csv"
        if not baseline_file.exists():
            continue
        
        df_base = pd.read_csv(baseline_file, dtype={"tract_geoid": str})
        df_base['tract'] = df_base['tract_geoid'].astype(str)
        df_base['tract_type'] = df_base['tract'].apply(
            lambda x: 'flood-prone' if x in flood_prone_tracts else 'non-prone'
        )
        
        # Baseline FI by tract/tenure
        base_fi = df_base.groupby(['tract', 'group', 'tract_type'])['has_FI'].mean().reset_index()
        base_fi.rename(columns={'has_FI': 'fi_baseline'}, inplace=True)
        
        # Load post-flood states for k=0,1,2
        for k in range(3):
            post_year = flood_year + k
            post_file = states_dir / f"state_for_vuln_{post_year}.csv"
            if not post_file.exists():
                continue
            
            df_post = pd.read_csv(post_file, dtype={"tract_geoid": str})
            df_post['tract'] = df_post['tract_geoid'].astype(str)
            df_post['tract_type'] = df_post['tract'].apply(
                lambda x: 'flood-prone' if x in flood_prone_tracts else 'non-prone'
            )
            
            post_fi = df_post.groupby(['tract', 'group', 'tract_type'])['has_FI'].mean().reset_index()
            post_fi.rename(columns={'has_FI': 'fi_post'}, inplace=True)
            
            # Merge and calculate delta
            merged = base_fi.merge(post_fi, on=['tract', 'group', 'tract_type'], how='inner')
            merged['fi_delta'] = merged['fi_post'] - merged['fi_baseline']
            merged['fi_delta_pct'] = merged['fi_delta'] * 100
            merged['flood_event'] = flood_year
            merged['k'] = k + 1  # k starts from 1 (year 1, 2, 3 after flood)
            merged['post_year'] = post_year
            merged.rename(columns={'group': 'tenure'}, inplace=True)
            
            all_rows.append(merged[['tract', 'tenure', 'tract_type', 'flood_event', 'k', 
                                    'fi_baseline', 'fi_post', 'fi_delta', 'fi_delta_pct']])
    
    if not all_rows:
        print("No data for FI takeup change analysis")
        return
    
    result_df = pd.concat(all_rows, ignore_index=True)
    
    # Create summary by tenure, tract_type, k (median across tracts)
    summary_rows = []
    for flood_event in result_df['flood_event'].unique():
        for tenure in ['owner', 'renter']:
            for tract_type in ['flood-prone', 'non-prone']:
                for k in [1, 2, 3]:  # k starts from 1 (year 1, 2, 3 after flood)
                    subset = result_df[
                        (result_df['flood_event'] == flood_event) &
                        (result_df['tenure'] == tenure) &
                        (result_df['tract_type'] == tract_type) &
                        (result_df['k'] == k)
                    ]
                    if len(subset) == 0:
                        continue
                    
                    # Remove outliers (1.5*IQR)
                    vals = subset['fi_delta_pct']
                    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                    iqr = q3 - q1
                    filtered = vals[(vals >= q1 - 1.5*iqr) & (vals <= q3 + 1.5*iqr)]
                    
                    summary_rows.append({
                        'flood_event': flood_event,
                        'tenure': tenure,
                        'tract_type': tract_type,
                        'k': k,
                        'n_tracts': len(filtered),
                        'fi_delta_median': filtered.median(),
                        'fi_delta_mean': filtered.mean(),
                        'fi_delta_std': filtered.std(),
                    })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Create cross-event average
    avg_rows = []
    for tenure in ['owner', 'renter']:
        for tract_type in ['flood-prone', 'non-prone']:
            for k in [0, 1, 2]:
                subset = summary_df[
                    (summary_df['tenure'] == tenure) &
                    (summary_df['tract_type'] == tract_type) &
                    (summary_df['k'] == k)
                ]
                if len(subset) == 0:
                    continue
                avg_rows.append({
                    'tenure': tenure,
                    'tract_type': tract_type,
                    'k': k,
                    'avg_delta_median': subset['fi_delta_median'].mean(),
                })
    
    avg_df = pd.DataFrame(avg_rows)
    
    # Create tract-level average across flood events
    tract_avg = result_df.groupby(['tract', 'tenure', 'tract_type', 'k']).agg({
        'fi_delta_pct': 'mean',
        'fi_baseline': 'mean',
        'fi_post': 'mean',
    }).reset_index()
    tract_avg.columns = ['tract', 'tenure', 'tract_type', 'k', 'avg_fi_delta_pct', 'avg_fi_baseline', 'avg_fi_post']
    
    # Export to Excel
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / 'fi_takeup_change.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='FI_by_Tract', index=False)
        tract_avg.to_excel(writer, sheet_name='FI_by_Tract_Avg', index=False)
        summary_df.to_excel(writer, sheet_name='Summary_by_Event', index=False)
        avg_df.to_excel(writer, sheet_name='Average_Across_Events', index=False)
    
    print(f"Saved: {excel_path}")


def export_all_excel(
    base_dir: Path,
    output_dir: Path = None,
    flood_prone_tracts: list[str] = None,
) -> None:
    """Export all Excel files to a dedicated folder.
    
    Args:
        base_dir: Base output directory (e.g., outputs/baseline/baseline)
        output_dir: Directory for Excel files (default: base_dir/excel)
        flood_prone_tracts: List of flood-prone tract GEOIDs
    """
    if output_dir is None:
        output_dir = base_dir / 'excel'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting Excel files to: {output_dir}")
    
    # TP analysis
    tp_path = base_dir / 'tp_traj.csv'
    if tp_path.exists():
        export_tp_excel(tp_path, output_dir, flood_prone_tracts)
    
    # Action analysis
    decisions_dir = base_dir / 'decisions'
    if decisions_dir.exists():
        export_action_excel(decisions_dir, output_dir, flood_prone_tracts)
        # Action rate delta analysis (all actions with baseline comparison)
        export_action_rate_delta(decisions_dir, output_dir, flood_prone_tracts)
    
    # FI takeup change analysis
    states_dir = base_dir / 'states'
    if states_dir.exists():
        export_fi_takeup_change(states_dir, output_dir, flood_prone_tracts)
    
    print("Excel export complete!")


def main():
    # Example usage
    base_dir = Path('outputs/baseline/baseline')
    
    # Define flood-prone tracts (can be loaded from config)
    # For now, use empty list (all tracts labeled as 'all')
    flood_prone_tracts = []
    
    export_all_excel(base_dir, flood_prone_tracts=flood_prone_tracts)


if __name__ == "__main__":
    main()
