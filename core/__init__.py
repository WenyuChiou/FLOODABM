# -*- coding: utf-8 -*-
"""
FLOODABM Core Package
=====================

This package contains the core simulation logic shared across all entry points:
- main.py (scenario analysis)
- sa_main.py (sensitivity analysis)
- main_mc.py (Monte Carlo stability testing)

Modules:
    config: Unified configuration loading and validation
    cli: Command-line argument parsing
"""
from pathlib import Path

from .config import ABMConfig, load_config, create_rng, resolve_data_path
from .cli import (
    create_scenario_parser,
    create_sensitivity_parser,
    create_montecarlo_parser,
    parse_args,
    args_to_config_overrides,
)

# Package root for relative imports
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

__all__ = [
    "PACKAGE_ROOT",
    "ABMConfig",
    "load_config",
    "create_rng",
    "resolve_data_path",
    "create_scenario_parser",
    "create_sensitivity_parser",
    "create_montecarlo_parser",
    "parse_args",
    "args_to_config_overrides",
]
