# -*- coding: utf-8 -*-
"""
CLI Argument Parser for FLOODABM
================================

Shared command-line argument parsing for all entry points.
This ensures consistent CLI interface across main.py, sa_main.py, and main_mc.py.

Example:
    >>> from core.cli import create_parser, parse_args
    >>> args = parse_args()
    >>> print(args.scenario)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def create_base_parser(description: str = "FLOODABM Simulation") -> argparse.ArgumentParser:
    """Create the base argument parser with common options.
    
    Args:
        description: Description text for the parser
        
    Returns:
        Configured ArgumentParser with common arguments
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Scenario selection
    parser.add_argument(
        "--scenario",
        choices=["baseline", "worst"],
        help="Scenario type: baseline (normal decisions) or worst (no-action)",
    )
    
    # Threshold overrides
    parser.add_argument(
        "--thr-mg",
        type=float,
        metavar="RATIO",
        help="Override flood ratio threshold for owner",
    )
    parser.add_argument(
        "--thr-nmg",
        type=float,
        metavar="RATIO",
        help="Override flood ratio threshold for renter",
    )
    
    # Output options
    parser.add_argument(
        "--out-root",
        type=str,
        metavar="PATH",
        help="Custom output root directory (default: <project>/outputs)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all visualization/plotting",
    )
    
    # Reproducibility
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use fixed RNG seed each year (no year offset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="Override random seed from config",
    )
    
    return parser


def create_scenario_parser() -> argparse.ArgumentParser:
    """Create parser for main.py (scenario analysis).
    
    Adds scenario-comparison specific arguments.
    """
    parser = create_base_parser("FLOODABM Scenario Analysis (Baseline vs Worst-Case)")
    
    # Comparison options
    parser.add_argument(
        "--compare-flood-or",
        action="store_true",
        help="Generate comparison plots for Owner & Renter groups",
    )
    parser.add_argument(
        "--compare-mode",
        choices=["total", "avg"],
        default="total",
        help="Comparison metric: total USD or average per household",
    )
    parser.add_argument(
        "--compare-severe-years",
        type=str,
        default="",
        metavar="YEARS",
        help="Comma-separated severe flood years to highlight (e.g., '2011,2014,2021')",
    )
    
    return parser


def create_sensitivity_parser() -> argparse.ArgumentParser:
    """Create parser for sa_main.py (sensitivity analysis).
    
    Adds sensitivity-specific arguments.
    """
    parser = create_base_parser("FLOODABM Sensitivity Analysis")
    
    parser.add_argument(
        "--mg-range",
        type=str,
        default="0.1,0.8",
        metavar="MIN,MAX",
        help="MG threshold range for sensitivity sweep (default: 0.1,0.8)",
    )
    parser.add_argument(
        "--nmg-range",
        type=str,
        default="0.1,0.8",
        metavar="MIN,MAX",
        help="NMG threshold range for sensitivity sweep (default: 0.1,0.8)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=8,
        metavar="N",
        help="Number of threshold steps per dimension (default: 8)",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing run outputs if available",
    )
    
    return parser


def create_montecarlo_parser() -> argparse.ArgumentParser:
    """Create parser for main_mc.py (Monte Carlo analysis).
    
    Adds Monte Carlo specific arguments.
    """
    parser = create_base_parser("FLOODABM Monte Carlo Stability Analysis")
    
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        metavar="N",
        help="Number of Monte Carlo runs (default: 10)",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=71001,
        metavar="N",
        help="Starting seed for initialization RNG (default: 71001)",
    )
    parser.add_argument(
        "--crn-seed",
        type=int,
        default=2025,
        metavar="N",
        help="Common random numbers seed (default: 2025)",
    )
    
    return parser


def parse_args(
    parser: Optional[argparse.ArgumentParser] = None,
    argv: Optional[list[str]] = None,
) -> argparse.Namespace:
    """Parse command-line arguments.
    
    Args:
        parser: ArgumentParser to use (default: create_base_parser())
        argv: Argument list (default: sys.argv)
        
    Returns:
        Parsed arguments namespace
    """
    if parser is None:
        parser = create_base_parser()
    return parser.parse_args(argv)


def args_to_config_overrides(args: argparse.Namespace) -> dict:
    """Convert parsed arguments to config override dictionary.
    
    Args:
        args: Parsed argument namespace
        
    Returns:
        Dictionary suitable for load_config(cli_overrides=...)
    """
    overrides = {}
    
    if getattr(args, "thr_mg", None) is not None:
        overrides["flood.ratio_threshold_MG"] = args.thr_mg
    
    if getattr(args, "thr_nmg", None) is not None:
        overrides["flood.ratio_threshold_NMG"] = args.thr_nmg
    
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    
    return overrides
