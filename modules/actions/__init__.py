# -*- coding: utf-8 -*-
"""
Actions Module
==============

Agent decision-making and psychology (Trust in Policymaker) updates.

Key Components:
    - bayes_fast_predictors: Fast Bayesian predictor loading (.pkl → .npz cache)
    - mgmix_pipeline: Main yearly simulation loop for MG/NMG decisions
    - mgmix_decision: Decision helper functions and state indexer
    - mgmix_tp: TP (Trust in Policymaker) update logic
    - vuln_for_tp: Flood ratio calculations from vulnerability data

Example:
    >>> from modules.actions import run_one_year_mgmix_fast
    >>> from modules.actions.bayes_fast_predictors import build_fast_predictors
"""

from .mgmix_pipeline import run_one_year_mgmix_fast, run_loop_mgmix_fast
from .mgmix_decision import load_predictors, build_state_indexer
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
    "mgmix_pipeline",
    "mgmix_decision",
    "mgmix_tp",
    "vuln_for_tp",
    "bayes_fast_predictors",
]
