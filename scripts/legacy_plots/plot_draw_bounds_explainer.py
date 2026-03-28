# -*- coding: utf-8 -*-
"""
Generate presentation figures explaining the bounded draw mechanism.
Figure 1: Conceptual comparison of standard vs bounded Bernoulli
Figure 2: Action-specific bounds with feasibility interpretation
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'figure.dpi': 200,
})

# ── Figure 1: Standard vs Bounded Bernoulli ──────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

p_vals = np.linspace(0, 1, 200)

# Left: Standard U(0,1)
ax = axes[0]
ax.fill_between(p_vals, 0, p_vals, color='#2196F3', alpha=0.3, label='Adopt (u < p)')
ax.fill_between(p_vals, p_vals, 1, color='#F44336', alpha=0.15, label='Reject (u ≥ p)')
ax.set_xlabel('Adoption probability p')
ax.set_ylabel('Uniform draw u')
ax.set_title('Standard Bernoulli: u ~ U(0, 1)')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
ax.legend(loc='upper left', fontsize=10)
ax.set_aspect('equal')

# Right: Bounded U(l, h) — use FI as example
ax = axes[1]
l_a, h_a = 0.35, 0.55

# Region: p < l → never adopt
ax.fill_betweenx([l_a, h_a], 0, l_a, color='#F44336', alpha=0.20)
ax.text(l_a/2 - 0.02, (l_a + h_a)/2, 'Never\nadopt', ha='center', va='center',
        fontsize=10, color='#C62828', fontweight='bold')

# Region: p > h → always adopt
ax.fill_betweenx([l_a, h_a], h_a, 1.0, color='#2196F3', alpha=0.30)
ax.text((h_a + 1.0)/2, (l_a + h_a)/2, 'Always\nadopt', ha='center', va='center',
        fontsize=10, color='#1565C0', fontweight='bold')

# Region: l < p < h → stochastic
p_mid = np.linspace(l_a, h_a, 50)
ax.fill_between(p_mid, l_a, p_mid, color='#2196F3', alpha=0.3)
ax.fill_between(p_mid, p_mid, h_a, color='#F44336', alpha=0.15)
ax.plot([l_a, h_a], [l_a, h_a], 'k--', lw=1.5, alpha=0.5)
ax.text((l_a + h_a)/2, h_a - 0.02, 'Stochastic\nzone', ha='center', va='top',
        fontsize=10, fontstyle='italic', color='#333333')

# Boundary lines
ax.axhline(l_a, color='gray', ls=':', lw=1)
ax.axhline(h_a, color='gray', ls=':', lw=1)
ax.axvline(l_a, color='gray', ls=':', lw=1)
ax.axvline(h_a, color='gray', ls=':', lw=1)

ax.text(-0.08, l_a, f'$l_a$ = {l_a}', ha='right', va='center', fontsize=10)
ax.text(-0.08, h_a, f'$h_a$ = {h_a}', ha='right', va='center', fontsize=10)

ax.set_xlabel('Adoption probability p (willingness)')
ax.set_ylabel('Stochastic threshold $u_a$ (feasibility)')
ax.set_title(f'Bounded Draw: $u_a$ ~ U({l_a}, {h_a})  [FI example]')
ax.set_xlim(-0.15, 1.05); ax.set_ylim(0.15, 0.75)

fig.tight_layout(w_pad=3)
out1 = 'draw_bounds_concept.png'
fig.savefig(out1, bbox_inches='tight')
print(f'Saved: {out1}')
plt.close()


# ── Figure 2: All actions — bounds comparison ────────────────────────

actions = ['FI', 'EH', 'BP', 'RL']
groups  = ['Both', 'Owner', 'Owner', 'Renter']
bounds  = [(0.35, 0.55), (0.30, 0.60), (0.25, 0.65), (0.25, 0.50)]
colors  = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
descs   = [
    'Standardized NFIP\napplication process',
    'Costs vary by building\ntype & foundation',
    'Multi-level govt approval\n& competitive funding',
    'Low financial barrier\nbut housing availability\nconstrains upper bound',
]

fig, ax = plt.subplots(figsize=(10, 5))

y_positions = np.arange(len(actions))[::-1]

for i, (act, grp, (lo, hi), col, desc) in enumerate(zip(actions, groups, bounds, colors, descs)):
    y = y_positions[i]
    width = hi - lo
    mid = (lo + hi) / 2

    # Bar
    ax.barh(y, width, left=lo, height=0.5, color=col, alpha=0.7, edgecolor=col, linewidth=1.5)

    # Bounds text
    ax.text(lo - 0.02, y, f'{lo:.2f}', ha='right', va='center', fontsize=11, fontweight='bold')
    ax.text(hi + 0.02, y, f'{hi:.2f}', ha='left', va='center', fontsize=11, fontweight='bold')

    # Width annotation
    ax.text(mid, y + 0.30, f'width = {width:.2f}', ha='center', va='bottom',
            fontsize=10, color=col, fontweight='bold')

    # Action label
    ax.text(-0.12, y, f'{act}\n({grp})', ha='center', va='center', fontsize=11, fontweight='bold')

    # Description
    ax.text(0.98, y, desc, ha='left', va='center', fontsize=9, color='#555555',
            fontstyle='italic')

# Annotations
ax.axvline(0.5, color='gray', ls='--', lw=1, alpha=0.4)
ax.text(0.5, y_positions[0] + 0.55, 'midpoint = 0.50', ha='center', va='bottom',
        fontsize=9, color='gray')

ax.set_xlim(-0.20, 1.45)
ax.set_ylim(-0.5, y_positions[0] + 0.8)
ax.set_xlabel('Stochastic threshold range [$l_a$, $h_a$]', fontsize=12)
ax.set_yticks([])
ax.set_title('Action-Specific Implementation Feasibility Bounds', fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Legend box
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8)
legend_text = ('Narrow interval → low implementation uncertainty\n'
               'Wide interval → high implementation uncertainty\n'
               'p > $h_a$ → certain adoption  |  p < $l_a$ → no adoption')
ax.text(0.55, -0.35, legend_text, fontsize=9, va='top', ha='center',
        bbox=props, transform=ax.transData)

fig.tight_layout()
out2 = 'draw_bounds_by_action.png'
fig.savefig(out2, bbox_inches='tight')
print(f'Saved: {out2}')
plt.close()
