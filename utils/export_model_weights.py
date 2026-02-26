# -*- coding: utf-8 -*-
"""
Model Weight Export Utility
===========================

Export Bayesian model weights from .pkl to .npz format for fast loading.

Usage:
    # From command line (in bayesian_engine environment):
    python -m utils.export_model_weights --model optimized
    
    # Or programmatically:
    from utils.export_model_weights import export_model_weights
    export_model_weights("models/optimized", "models/optimized_fast")
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from typing import Optional

# Models to export
MODELS = ["owner_FI", "owner_EH", "owner_BP", "owner_RL", "renter_FI", "renter_EH", "renter_BP", "renter_RL"]


def export_model_weights(
    src_dir: Path | str,
    dst_dir: Optional[Path | str] = None,
    verbose: bool = True
) -> int:
    """
    Export Bayesian model weights from pkl to npz format.
    
    Args:
        src_dir: Source directory containing .pkl files
        dst_dir: Destination directory for .npz files (default: src_dir + "_fast")
        verbose: Print progress messages
        
    Returns:
        Number of models successfully exported
    """
    try:
        import dill
    except ImportError:
        raise ImportError("dill is required: pip install dill")
    
    src_dir = Path(src_dir)
    if dst_dir is None:
        dst_dir = Path(str(src_dir) + "_fast")
    else:
        dst_dir = Path(dst_dir)
    
    dst_dir.mkdir(exist_ok=True, parents=True)
    
    if verbose:
        print(f"Source: {src_dir}")
        print(f"Destination: {dst_dir}")
        print("=" * 60)
    
    success_count = 0
    for name in MODELS:
        pkl_path = src_dir / f"{name}.pkl"
        if not pkl_path.exists():
            if verbose:
                print(f"[SKIP] {name}: pkl not found")
            continue
        
        try:
            # Load pickle
            with open(pkl_path, 'rb') as f:
                obj = dill.load(f)
            
            # Extract weights from posterior samples
            ps = obj.posterior_samples
            
            # Try different key names for weights
            weights = None
            for key in ['beta', 'w', 'weights', 'coef']:
                if key in ps:
                    weights = np.mean(np.asarray(ps[key]), axis=0)
                    break
            
            if weights is None:
                if verbose:
                    print(f"[WARN] {name}: No weights found in keys: {list(ps.keys())}")
                weights = np.zeros(3, dtype=np.float32)
            
            # Try different key names for bias
            bias = 0.0
            for key in ['intercept', 'alpha', 'b', 'bias']:
                if key in ps:
                    bias = float(np.mean(np.asarray(ps[key])))
                    break
            
            # Ensure weights is 1D array of length 3
            weights = np.asarray(weights).flatten()
            if len(weights) < 3:
                weights = np.pad(weights, (0, 3 - len(weights)))
            elif len(weights) > 3:
                weights = weights[:3]
            
            # Save as npz
            npz_path = dst_dir / f"{name}.npz"
            np.savez(npz_path, w=weights.astype(np.float32), b=np.float32(bias))
            
            if verbose:
                print(f"[OK] {name}: w={weights.round(4)}, b={bias:.4f}")
            success_count += 1
            
        except Exception as e:
            if verbose:
                print(f"[ERROR] {name}: {e}")
    
    if verbose:
        print("=" * 60)
        print(f"Exported {success_count}/{len(MODELS)} models")
    
    return success_count


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Export Bayesian model weights to npz format")
    parser.add_argument("--model", "-m", required=True, help="Model name (e.g., optimized, baseline)")
    parser.add_argument("--src", "-s", help="Source directory (default: models/<model>)")
    parser.add_argument("--dst", "-d", help="Destination directory (default: models/<model>_fast)")
    
    args = parser.parse_args()
    
    src = Path(args.src) if args.src else Path("models") / args.model
    dst = Path(args.dst) if args.dst else None
    
    export_model_weights(src, dst)


if __name__ == "__main__":
    main()
