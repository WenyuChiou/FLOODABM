# -*- coding: utf-8 -*-
"""
FLOODABM Modules Package
========================

Core simulation modules for the Flood Agent-Based Model.

Subpackages:
    actions: Agent decision-making and psychology (TP) updates
    finance: Insurance premium and payout calculations
    vulnerability: Flood damage estimation

Example:
    >>> from modules.actions import run_one_year_mgmix_fast
    >>> from modules.finance import run_finance_for_year
"""

__all__ = ["actions", "finance", "vulnerability"]
__version__ = "0.2.0"
