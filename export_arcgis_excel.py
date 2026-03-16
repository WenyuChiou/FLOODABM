# -*- coding: utf-8 -*-
"""Export comprehensive tract-level data to Excel for ArcGIS Pro."""
import sys, json, yaml, re, glob
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path('outputs/baseline/baseline')
WORST = Path('outputs/baseline/worst')

# ── Load config ──
with open('config/overall_md_mean_by_tract_2011_2023.json', encoding='utf-8') as f:
    depths_raw = json.load(f)

depth_info = {}
for rec in depths_raw:
    t = str(rec['CensusTract'])
    vals = [rec[c] for c in rec if c.endswith('_mean') and rec[c] > 0]
    depth_info[t] = {
        'n_flood_years': len(vals),
        'mean_depth_m': round(sum(vals)/len(vals), 4) if vals else 0,
        'max_depth_m': round(max(vals), 4) if vals else 0,
    }
    for c in rec:
        if c.endswith('_mean'):
            yr = 2000 + int(c.split('_')[0])
            depth_info[t][f'depth_{yr}'] = rec[c]

fp_tracts = set(t for t, v in depth_info.items() if v['n_flood_years'] >= 7)

def get_county(t):
    return {'34013': 'Essex', '34027': 'Morris', '34031': 'Passaic'}.get(t[:5], t[:5])

all_tracts = sorted(depth_info.keys())

# ── Load baseline decisions ──
dec_b = pd.read_csv(BASE / 'decisions/action_share_owner_renter_tract_all_years.csv',
                     dtype={'tract_geoid': str})

# ── Load finance (per year) ──
def load_finance(root):
    frames = []
    for f in sorted(glob.glob(str(root / 'finance/finance_tract_*.csv'))):
        m = re.search(r'tract_(\d{4})\.csv', f)
        if m:
            df = pd.read_csv(f, dtype={'tract_geoid': str})
            df['year'] = int(m.group(1))
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

fin_b = load_finance(BASE)
fin_w = load_finance(WORST)

# ── Load damage ──
dmg_b = pd.read_csv(BASE / 'vulnerability/flood_damage/flood_damage_tract_ALL_years.csv',
                     dtype={'tract_geoid': str})
dmg_w = pd.read_csv(WORST / 'vulnerability/flood_damage/flood_damage_tract_ALL_years.csv',
                     dtype={'tract_geoid': str})

# ══════════════════════════════════════════════════════════
# Sheet 1: Tract Classification
# ══════════════════════════════════════════════════════════
rows1 = []
for tract in all_tracts:
    d = depth_info[tract]
    r = {'tract_geoid': tract, 'county': get_county(tract),
         'flood_prone': 1 if tract in fp_tracts else 0,
         'flood_prone_label': 'Flood-Prone' if tract in fp_tracts else 'Non-Flood-Prone',
         'n_flood_years': d['n_flood_years'],
         'mean_depth_m': d['mean_depth_m'], 'max_depth_m': d['max_depth_m']}
    for yr in range(2011, 2024):
        r[f'depth_{yr}_m'] = d.get(f'depth_{yr}', 0)
    rows1.append(r)
df_class = pd.DataFrame(rows1)

# ══════════════════════════════════════════════════════════
# Sheet 2: Adaptation Rates (all actions, long format)
# ══════════════════════════════════════════════════════════
rows2 = []
for _, row in dec_b.iterrows():
    t = row['tract_geoid']
    yr = int(row['year'])
    r = {'tract_geoid': t, 'county': get_county(t),
         'flood_prone': 1 if t in fp_tracts else 0, 'year': yr}
    for col_prefix, label in [
        ('owner_share_FI', 'owner_FI'), ('owner_share_EH', 'owner_EH'),
        ('owner_share_BP', 'owner_BP'), ('owner_share_DN', 'owner_DN'),
        ('renter_share_FI', 'renter_FI'), ('renter_share_RL', 'renter_RL'),
        ('renter_share_DN', 'renter_DN'),
    ]:
        if col_prefix in row.index:
            r[f'{label}_pct'] = round(row[col_prefix] * 100, 2)
    r['owner_N'] = row.get('owner_N_total', np.nan)
    r['renter_N'] = row.get('renter_N_total', np.nan)
    rows2.append(r)
df_actions = pd.DataFrame(rows2)

# ══════════════════════════════════════════════════════════
# Sheet 3: Finance (baseline + worst, long format)
# ══════════════════════════════════════════════════════════
fin_cols = ['gross_total_kUSD', 'owner_gross_total_kUSD', 'renter_gross_total_kUSD',
            'payout_total_kUSD', 'oop_total_kUSD', 'premium_total_kUSD',
            'policyholders', 'claims', 'households']
rows3 = []
for scenario, fin_df in [('baseline', fin_b), ('worst', fin_w)]:
    for _, row in fin_df.iterrows():
        t = str(row['tract_geoid'])
        r = {'tract_geoid': t, 'county': get_county(t),
             'flood_prone': 1 if t in fp_tracts else 0,
             'year': int(row['year']), 'scenario': scenario}
        for c in fin_cols:
            if c in row.index:
                r[c] = row[c]
        rows3.append(r)
df_finance = pd.DataFrame(rows3)

# ══════════════════════════════════════════════════════════
# Sheet 4: Damage (baseline + worst, long format)
# ══════════════════════════════════════════════════════════
rows4 = []
for scenario, dmg_df in [('baseline', dmg_b), ('worst', dmg_w)]:
    for _, row in dmg_df.iterrows():
        t = str(row['tract_geoid'])
        rows4.append({
            'tract_geoid': t, 'county': get_county(t),
            'flood_prone': 1 if t in fp_tracts else 0,
            'year': int(row['year']), 'scenario': scenario,
            'owner_damage_usd': row.get('owner_usd', np.nan),
            'renter_damage_usd': row.get('renter_usd', np.nan),
            'total_damage_usd': row.get('both_usd', np.nan),
        })
df_damage = pd.DataFrame(rows4)

# ══════════════════════════════════════════════════════════
# Sheet 5: Pre/Post Flood Delta (all actions)
# ══════════════════════════════════════════════════════════
events = [(2013, 2014, '2014_flood'), (2020, 2021, '2021_flood')]
action_map = {'owner': ['FI', 'EH', 'BP'], 'renter': ['FI', 'RL']}
rows5 = []
for tract in all_tracts:
    for base_yr, event_yr, event_label in events:
        for tenure, actions in action_map.items():
            for action in actions:
                col = f'{tenure}_share_{action}'
                if col not in dec_b.columns:
                    continue
                pre = dec_b[(dec_b.tract_geoid == tract) & (dec_b.year == base_yr)]
                if pre.empty:
                    continue
                pre_v = pre[col].values[0] * 100
                r = {'tract_geoid': tract, 'county': get_county(tract),
                     'flood_prone': 1 if tract in fp_tracts else 0,
                     'flood_event': event_label, 'tenure': tenure, 'action': action,
                     'pre_flood_rate_pct': round(pre_v, 2)}
                for k, label in [(0, 'flood_yr'), (1, 'plus1yr'), (2, 'plus2yr')]:
                    yr = event_yr + k
                    post = dec_b[(dec_b.tract_geoid == tract) & (dec_b.year == yr)]
                    if not post.empty:
                        post_v = post[col].values[0] * 100
                        r[f'{label}_rate_pct'] = round(post_v, 2)
                        r[f'{label}_delta_pp'] = round(post_v - pre_v, 2)
                rows5.append(r)
df_delta = pd.DataFrame(rows5)

# ══════════════════════════════════════════════════════════
# Sheet 6: ArcGIS Wide Join (one row per tract, everything)
# ══════════════════════════════════════════════════════════
rows6 = []
for tract in all_tracts:
    d = depth_info.get(tract, {})
    r = {'tract_geoid': tract, 'county': get_county(tract),
         'flood_prone': 1 if tract in fp_tracts else 0,
         'mean_depth_m': d.get('mean_depth_m', 0),
         'n_flood_yrs': d.get('n_flood_years', 0)}

    # All action rates by year
    for yr in range(2011, 2024):
        sub = dec_b[(dec_b.tract_geoid == tract) & (dec_b.year == yr)]
        if sub.empty:
            continue
        s = sub.iloc[0]
        r[f'oFI_{yr}'] = round(s.get('owner_share_FI', 0) * 100, 2)
        r[f'oEH_{yr}'] = round(s.get('owner_share_EH', 0) * 100, 2)
        r[f'oBP_{yr}'] = round(s.get('owner_share_BP', 0) * 100, 2)
        r[f'rFI_{yr}'] = round(s.get('renter_share_FI', 0) * 100, 2)
        r[f'rRL_{yr}'] = round(s.get('renter_share_RL', 0) * 100, 2)

    # Cumulative damage & finance (baseline)
    sub_dmg = dmg_b[dmg_b.tract_geoid == tract]
    r['b_cum_dmg_kUSD'] = round(sub_dmg['both_usd'].sum() / 1000, 1) if not sub_dmg.empty else 0
    r['b_o_cum_dmg_kUSD'] = round(sub_dmg['owner_usd'].sum() / 1000, 1) if not sub_dmg.empty else 0
    r['b_r_cum_dmg_kUSD'] = round(sub_dmg['renter_usd'].sum() / 1000, 1) if not sub_dmg.empty else 0

    sub_fin = fin_b[fin_b.tract_geoid == tract]
    r['b_cum_payout_kUSD'] = round(sub_fin['payout_total_kUSD'].sum(), 1) if not sub_fin.empty else 0
    r['b_cum_oop_kUSD'] = round(sub_fin['oop_total_kUSD'].sum(), 1) if not sub_fin.empty else 0
    r['b_cum_prem_kUSD'] = round(sub_fin['premium_total_kUSD'].sum(), 1) if not sub_fin.empty else 0

    # Cumulative damage & finance (worst)
    sub_dmg_w = dmg_w[dmg_w.tract_geoid == tract]
    r['w_cum_dmg_kUSD'] = round(sub_dmg_w['both_usd'].sum() / 1000, 1) if not sub_dmg_w.empty else 0
    r['w_o_cum_dmg_kUSD'] = round(sub_dmg_w['owner_usd'].sum() / 1000, 1) if not sub_dmg_w.empty else 0
    r['w_r_cum_dmg_kUSD'] = round(sub_dmg_w['renter_usd'].sum() / 1000, 1) if not sub_dmg_w.empty else 0

    sub_fin_w = fin_w[fin_w.tract_geoid == tract]
    r['w_cum_payout_kUSD'] = round(sub_fin_w['payout_total_kUSD'].sum(), 1) if not sub_fin_w.empty else 0
    r['w_cum_oop_kUSD'] = round(sub_fin_w['oop_total_kUSD'].sum(), 1) if not sub_fin_w.empty else 0

    # Damage reduction (baseline vs worst)
    if r['w_cum_dmg_kUSD'] > 0:
        r['dmg_reduce_pct'] = round((1 - r['b_cum_dmg_kUSD'] / r['w_cum_dmg_kUSD']) * 100, 1)

    # Per-household metrics
    hh_sub = dec_b[(dec_b.tract_geoid == tract) & (dec_b.year == 2023)]
    if not hh_sub.empty:
        total_hh = hh_sub['owner_N_total'].values[0] + hh_sub['renter_N_total'].values[0]
        if total_hh > 0:
            r['b_dmg_per_hh_kUSD'] = round(r['b_cum_dmg_kUSD'] / total_hh, 2)
            r['w_dmg_per_hh_kUSD'] = round(r['w_cum_dmg_kUSD'] / total_hh, 2)

    # Deltas for all actions
    for base_yr, event_yr, prefix in [(2013, 2014, 'e14'), (2020, 2021, 'e21')]:
        for tenure_short, pairs in [('o', [('owner', 'FI'), ('owner', 'EH'), ('owner', 'BP')]),
                                     ('r', [('renter', 'FI'), ('renter', 'RL')])]:
            for ten_full, act in pairs:
                col = f'{ten_full}_share_{act}'
                if col not in dec_b.columns:
                    continue
                pre = dec_b[(dec_b.tract_geoid == tract) & (dec_b.year == base_yr)]
                if pre.empty:
                    continue
                pre_v = pre[col].values[0] * 100
                for k, suf in [(0, 't0'), (1, 't1'), (2, 't2')]:
                    post = dec_b[(dec_b.tract_geoid == tract) & (dec_b.year == event_yr + k)]
                    if not post.empty:
                        r[f'{tenure_short}{act}d_{prefix}_{suf}'] = round(
                            post[col].values[0] * 100 - pre_v, 2)
    rows6.append(r)
df_wide = pd.DataFrame(rows6)

# ══════════════════════════════════════════════════════════
# Write Excel
# ══════════════════════════════════════════════════════════
out_path = BASE / 'visualization/all_tract_data_arcgis.xlsx'
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    df_class.to_excel(writer, sheet_name='1_Tract_Classification', index=False)
    df_actions.to_excel(writer, sheet_name='2_Adaptation_Rates', index=False)
    df_finance.to_excel(writer, sheet_name='3_Finance', index=False)
    df_damage.to_excel(writer, sheet_name='4_Damage', index=False)
    df_delta.to_excel(writer, sheet_name='5_PrePost_Delta', index=False)
    df_wide.to_excel(writer, sheet_name='6_ArcGIS_Wide_Join', index=False)

print(f'Saved: {out_path}')
print(f'  1_Tract_Classification: {df_class.shape}')
print(f'  2_Adaptation_Rates:     {df_actions.shape}')
print(f'  3_Finance:              {df_finance.shape}')
print(f'  4_Damage:               {df_damage.shape}')
print(f'  5_PrePost_Delta:        {df_delta.shape}')
print(f'  6_ArcGIS_Wide_Join:     {df_wide.shape}')
