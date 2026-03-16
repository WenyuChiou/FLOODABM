"""
Posterior Predictive Query Tool
================================
Given (TP, CP, SP) values, show the full posterior distribution of adoption probability.
Usage:
    python posterior_predictive_query.py --tp 0.6 --cp 0.7 --sp 0.5
    python posterior_predictive_query.py --tp 0.6 --cp 0.7 --sp 0.5 --model owner_FI
    python posterior_predictive_query.py --scenarios   # pre-defined scenario comparison
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
ALL_MODELS = {
    'owner_FI':  'Owner: Flood Insurance',
    'owner_EH':  'Owner: Elevation/Hardening',
    'owner_BP':  'Owner: Buyout Program',
    'renter_FI': 'Renter: Flood Insurance',
    'renter_RL': 'Renter: Relocation',
}

def load_posterior(model_name):
    npz = np.load(MODELS_DIR / f"{model_name}.npz", allow_pickle=True)
    return {
        'b0':  npz['posterior_beta_0'],
        'btp': npz['posterior_beta_tp'],
        'bcp': npz['posterior_beta_cp'],
        'bsp': npz['posterior_beta_sp'],
        'phi': npz['posterior_phi'],
    }

def predict_mu_distribution(post, tp, cp, sp):
    """Returns 4800 posterior draws of mu."""
    linear = post['b0'] + post['btp']*tp + post['bcp']*cp + post['bsp']*sp
    return expit(linear)

def plot_single_query(tp, cp, sp, model_name=None, out_path=None):
    """Plot posterior predictive for one (TP,CP,SP) across all or one model."""
    models = {model_name: ALL_MODELS[model_name]} if model_name else ALL_MODELS
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5))
    if n == 1:
        axes = [axes]
    
    for ax, (mname, mtitle) in zip(axes, models.items()):
        post = load_posterior(mname)
        mu_draws = predict_mu_distribution(post, tp, cp, sp)
        
        # Histogram + stats
        ax.hist(mu_draws, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='Threshold (0.5)')
        ax.axvline(mu_draws.mean(), color='blue', linewidth=2, label=f'Mean={mu_draws.mean():.3f}')
        
        q05, q95 = np.percentile(mu_draws, [5, 95])
        ax.axvspan(q05, q95, alpha=0.1, color='blue', label=f'90% CI [{q05:.3f}, {q95:.3f}]')
        
        adopt_pct = (mu_draws > 0.5).mean() * 100
        ax.set_title(f'{mtitle}\nP(adopt)={mu_draws.mean():.3f}, P(μ>0.5)={adopt_pct:.0f}%',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Adoption Probability (μ)')
        ax.set_ylabel('Density')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Posterior Predictive Distribution\nInput: TP={tp:.2f}, CP={cp:.2f}, SP={sp:.2f}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if out_path is None:
        out_path = Path(__file__).parent / f'query_TP{tp:.2f}_CP{cp:.2f}_SP{sp:.2f}.pdf'
    fig.savefig(out_path, dpi=300)
    fig.savefig(str(out_path).replace('.pdf', '.png'), dpi=300)
    plt.close(fig)
    print(f'Saved: {out_path}')
    return out_path

def plot_scenarios():
    """Pre-defined scenario comparison for reviewer."""
    scenarios = [
        (0.2, 0.5, 0.5, 'Low TP (post-calm)'),
        (0.5, 0.5, 0.5, 'Medium TP (baseline)'),
        (0.8, 0.5, 0.5, 'High TP (post-flood)'),
        (0.8, 0.8, 0.8, 'High all (engaged)'),
        (0.8, 0.2, 0.2, 'High TP, Low CP/SP (fearful but helpless)'),
    ]
    
    fig, axes = plt.subplots(5, 5, figsize=(22, 20))
    
    for row, (tp, cp, sp, label) in enumerate(scenarios):
        for col, (mname, mtitle) in enumerate(ALL_MODELS.items()):
            ax = axes[row, col]
            post = load_posterior(mname)
            mu_draws = predict_mu_distribution(post, tp, cp, sp)
            
            ax.hist(mu_draws, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='white')
            ax.axvline(0.5, color='red', linestyle='--', linewidth=1)
            
            mean_mu = mu_draws.mean()
            adopt_pct = (mu_draws > 0.5).mean() * 100
            ax.axvline(mean_mu, color='blue', linewidth=1.5)
            
            ax.text(0.03, 0.95, f'μ={mean_mu:.2f}\nP(>0.5)={adopt_pct:.0f}%',
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.2)
            
            if row == 0:
                ax.set_title(mtitle.split(': ')[1], fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{label}\n(TP={tp}, CP={cp}, SP={sp})', fontsize=8)
            if row < 4:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('μ', fontsize=9)
    
    fig.suptitle('Posterior Predictive Distributions: 5 Scenarios × 5 Actions\n'
                 '(Blue line = posterior mean, Red dashed = 0.5 threshold)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = Path(__file__).parent / 'scenario_comparison_5x5.pdf'
    fig.savefig(out, dpi=300)
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=300)
    plt.close(fig)
    print(f'Saved: {out}')

    # Print summary table
    print('\n=== Scenario Summary (mean μ | P(μ>0.5)) ===')
    header = f'{"Scenario":<35}' + ''.join(f'{m:<15}' for m in ALL_MODELS.values())
    print(header)
    print('-' * len(header))
    for tp, cp, sp, label in scenarios:
        row_str = f'{label:<35}'
        for mname in ALL_MODELS:
            post = load_posterior(mname)
            mu = predict_mu_distribution(post, tp, cp, sp)
            row_str += f'{mu.mean():.2f} ({(mu>0.5).mean()*100:.0f}%)     '
        print(row_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Posterior Predictive Query')
    parser.add_argument('--tp', type=float, help='TP value (0-1)')
    parser.add_argument('--cp', type=float, help='CP value (0-1)')
    parser.add_argument('--sp', type=float, help='SP value (0-1)')
    parser.add_argument('--model', type=str, default=None, help='Specific model name')
    parser.add_argument('--scenarios', action='store_true', help='Run pre-defined scenarios')
    args = parser.parse_args()
    
    if args.scenarios:
        plot_scenarios()
    elif args.tp is not None and args.cp is not None and args.sp is not None:
        plot_single_query(args.tp, args.cp, args.sp, args.model)
    else:
        parser.print_help()
