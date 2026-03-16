# -*- coding: utf-8 -*-
"""
Actions Module
==============

Agent decision-making and psychology (Threat Perception) updates.

Key Components:
    - bayes_fast_predictors: Fast Bayesian predictor loading (.pkl → .npz cache)
    - pipeline: Main yearly simulation loop for owner/renter decisions
    - decision: Decision helper functions and state indexer
    - tp: Threat Perception (TP) update logic
    - vuln_for_tp: Flood ratio calculations from vulnerability data

Example:
    >>> from modules.actions import run_one_year_fast
    >>> from modules.actions.bayes_fast_predictors import build_fast_predictors
"""

from .pipeline import run_one_year_mgmix_fast, run_loop_mgmix_fast
from .decision import load_predictors, build_state_indexer
from .bayes_fast_predictors import build_fast_predictors

__all__ = [
    # Main entry points
    "run_one_year_mgmix_fast",
    "run_loop_mgmix_fast",

    # Predictor loading
    "load_predictors",
    "build_fast_predictors",

    # State management
    "build_state_indexer",

    # Submodules
    "pipeline",
    "decision",
    "tp",
    "vuln_for_tp",
    "bayes_fast_predictors",
]
