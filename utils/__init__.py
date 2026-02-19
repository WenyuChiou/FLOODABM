"""Utilities for FLOODABM.

Lightweight package init to avoid eager imports of heavy modules (e.g., matplotlib).
Exposes modules lazily so ``import utils`` doesn't require plotting libraries
until you actually access ``utils.plots``.
"""

from __future__ import annotations
from importlib import import_module
from typing import Any

__all__ = [
	"plots",
	"helpers",
	"config_loader",
	"plots_comparision_scenario",
	"main_helpers",
]


def __getattr__(name: str) -> Any:  # PEP 562 — lazy attribute access on module
	if name == "plots":
		return import_module(".plots", __name__)
	if name == "helpers":
		return import_module(".\u005fhelpers", __name__)  # alias _helpers as helpers
	if name == "config_loader":
		return import_module(".config_loader", __name__)
	if name == "plots_comparision_scenario":
		return import_module(".plots_comparision_scenario", __name__)
	if name == "main_helpers":
		return import_module(".main_helpers", __name__)
	raise AttributeError(name)


def __dir__():  # improve tab-complete experience
	return sorted(list(globals().keys()) + __all__)