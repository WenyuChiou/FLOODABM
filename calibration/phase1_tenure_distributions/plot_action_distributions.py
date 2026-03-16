"""
Plot adaptation action Likert histograms for Owner vs Renter.
- Owner actions: FI (Q24), EH (Q26), BP (Q28)
- Renter actions: FI (Q24), RL (Q30)
No Beta distribution fitting — actions are discrete Likert 1-5.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\Calibration\Basyiean regression mode\data\data_ori.xlsx")
OUT_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\Calibration\tenure_distributions")

owner = pd.read_excel(DATA_FILE, sheet_name='owner_variable', engine='openpyxl')
renter = pd.read_excel(DATA_FILE, sheet_name='renter_variable', engine='openpyxl')

print(f"Owner: {len(owner)} rows, Renter: {len(renter)} rows, Total: {len(owner)+len(renter)}")

ACTION_LABELS = {
    'FI': 'Flood Insurance (FI)',
    'EH': 'Elevation/Hardening (EH)',
    'BP': 'Buyout Program (BP)',
    'RL': 'Relocation (RL)',
}
ACTION_QCOL = {
    'FI': 'Q24', 'EH': 'Q26', 'BP': 'Q28', 'RL': 'Q30',
}
COLORS = {"owner": "#1f77b4", "renter": "#d62728"}
LIKERT_LABELS = ['Str. Disagree', 'Disagree', 'Neutral', 'Agree', 'Str. Agree']

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
    "legend.fontsize": 9, "lines.linewidth": 2.0,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})

# ════════════════════════════════════════════════════════════════
# Action Likert Histograms (5 panels)
# ════════════════════════════════════════════════════════════════
plot_list = [
    ('FI', 'owner', owner), ('FI', 'renter', renter),
    ('EH', 'owner', owner), ('BP', 'owner', owner),
    ('RL', 'renter', renter),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes_flat = axes.flatten()

for idx, (act, grp, df) in enumerate(plot_list):
    ax = axes_flat[idx]
    color = COLORS[grp]
    vals = df[act].dropna()
    n_resp = len(vals)
    n_total = len(df)

    counts = np.array([(vals == lk).sum() for lk in range(1, 6)])
    pcts = counts / n_resp * 100
    x_pos = np.arange(5)

    bars = ax.bar(x_pos, pcts, 0.6, color=color, alpha=0.8, edgecolor='white')
    for bar, cnt, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{int(cnt)}\n({pct:.1f}%)', ha='center', va='bottom',
                fontsize=7, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{i+1}\n{LIKERT_LABELS[i]}' for i in range(5)], fontsize=7)
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f"{ACTION_LABELS[act]} [{ACTION_QCOL[act]}]\n{grp.capitalize()} (n={n_resp})",
                 fontsize=11, fontweight='bold')
    ax.text(0.02, 0.95,
            f"mean={vals.mean():.2f}\nstd={vals.std():.2f}\nadopt(\u22654)={(vals >= 4).sum()/n_resp*100:.1f}%",
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax.grid(True, alpha=0.3, axis='y')

axes_flat[5].set_visible(False)

n_total = len(owner) + len(renter)
fig.suptitle(f'Adaptation Action Likert Histograms (n={n_total})\n'
             'Owner: FI, EH, BP | Renter: FI, RL',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / 'action_likert_histograms.png')
fig.savefig(OUT_DIR / 'action_likert_histograms.pdf')
plt.close(fig)
print("Saved: action_likert_histograms.png/pdf")

# ── Print summary ──
print("\n" + "=" * 60)
print("Action Distribution Summary")
print("=" * 60)
print(f"{'Group':<8} {'Action':<5} {'n':<6} {'mean':<6} {'std':<6} {'adopt%':<8}")
print("-" * 45)
for gname, df, actions in [('Owner', owner, ['FI','EH','BP']), ('Renter', renter, ['FI','RL'])]:
    for act in actions:
        vals = df[act].dropna()
        adopt = (vals >= 4).sum() / len(vals) * 100
        print(f"{gname:<8} {act:<5} {len(vals):<6} {vals.mean():<6.2f} {vals.std():<6.2f} {adopt:<8.1f}")
print(f"\nAll saved to: {OUT_DIR}")
