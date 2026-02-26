# -*- coding: utf-8 -*-
"""
Unified Configuration Module for FLOODABM
==========================================

This module consolidates configuration loading from multiple sources:
1. YAML config file (abm_params.yaml)
2. Command-line arguments
3. Environment variables

It provides dataclasses for type-safe configuration access and validation.

Example:
    >>> from core.config import load_config
    >>> cfg = load_config(config_path="config/abm_params.yaml", scenario="baseline")
    >>> print(cfg.seed, cfg.years)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


# =============================================================================
# Path Configuration
# =============================================================================

def get_project_root() -> Path:
    """Get the FLOODABM project root directory."""
    return Path(__file__).resolve().parent.parent


def get_default_config_path() -> Path:
    """Get the default configuration file path."""
    return get_project_root() / "config" / "abm_params.yaml"


# =============================================================================
# YAML Loading
# =============================================================================

def load_yaml(path: Path) -> dict:
    """Load and parse a YAML configuration file.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        Dictionary containing the parsed YAML content
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class FloodConfig:
    """Flood event thresholds and parameters."""
    ratio_threshold: float = 0.001
    ratio_threshold_owner: float = 0.5
    ratio_threshold_renter: float = 0.5
    depth_threshold_m: Optional[float] = None
    FFE_ft: float = 0.5


@dataclass
class TPConfig:
    """Trust in Policymaker (TP) update configuration."""
    shock_scale_owner: float = 1.0
    shock_scale_renter: float = 1.0
    shock_timing: str = "start"
    shock_mode: str = "additive"
    shock_min_ratio: float = 0.001
    clip_lo: float = 0.0
    clip_hi: float = 1.0


@dataclass
class ActionDynamicsConfig:
    """Agent action dynamics configuration."""
    eh_one_time: bool = False
    eh_once_min_ft: float = 3.0
    eh_once_max_ft: float = 5.0
    eh_step_ft: float = 1.0
    eh_cap_ft: float = 8.0
    apply_RL: bool = True
    rl_backfill: bool = True
    apply_BP: bool = True
    bp_backfill: bool = False
    bp_sampler: str = "lognorm_from_totals"


@dataclass
class OutputConfig:
    """Output and plotting configuration."""
    enable_plots: bool = True
    output_mode: str = "full"  # "full", "minimal", "essential"


@dataclass
class FilesConfig:
    """Data file paths configuration."""
    depths_overall: Optional[str] = None
    flood_events: Optional[str] = None
    households: Optional[str] = None


@dataclass
class ABMConfig:
    """Main ABM configuration container.
    
    This is the primary configuration object that holds all simulation parameters.
    It is created by the `load_config()` function.
    """
    # Core parameters
    seed: int = 12345
    years_step: float = 1.0
    scenario: str = "baseline"
    years: list[int] = field(default_factory=list)
    
    # Sub-configurations
    flood: FloodConfig = field(default_factory=FloodConfig)
    tp_config: TPConfig = field(default_factory=TPConfig)
    action_dynamics: ActionDynamicsConfig = field(default_factory=ActionDynamicsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    files: FilesConfig = field(default_factory=FilesConfig)
    
    # Paths (resolved at load time)
    project_root: Path = field(default_factory=get_project_root)
    config_path: Optional[Path] = None
    output_root: Optional[Path] = None
    
    # Raw YAML for advanced access
    _raw: dict = field(default_factory=dict, repr=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the raw config dict (for backward compatibility)."""
        return self._raw.get(key, default)
    
    @property
    def is_worst_case(self) -> bool:
        """Check if this is a worst-case (no-action) scenario."""
        return self.scenario.lower() == "worst"


# =============================================================================
# Configuration Loading
# =============================================================================

def _parse_flood_config(raw: dict) -> FloodConfig:
    """Parse flood configuration from raw YAML dict."""
    flood = raw.get("flood", {}) or {}
    single_thr = float(flood.get("ratio_threshold", 0.5))
    
    return FloodConfig(
        ratio_threshold=float(flood.get("ratio_threshold", 0.001)),
        ratio_threshold_owner=float(flood.get("ratio_threshold_owner", single_thr)),
        ratio_threshold_renter=float(flood.get("ratio_threshold_renter", single_thr)),
        depth_threshold_m=flood.get("depth_threshold_m"),
        FFE_ft=float(flood.get("FFE_ft", 0.5)),
    )


def _parse_tp_config(raw: dict) -> TPConfig:
    """Parse TP configuration from raw YAML dict."""
    tp = raw.get("tp_config", {}) or {}
    flood = raw.get("flood", {}) or {}
    
    return TPConfig(
        shock_scale_owner=float(tp.get("shock_scale_owner", 1.0)),
        shock_scale_renter=float(tp.get("shock_scale_renter", 1.0)),
        shock_timing=tp.get("shock_timing", "start"),
        shock_mode=tp.get("shock_mode", "additive"),
        shock_min_ratio=float(tp.get("shock_min_ratio", flood.get("ratio_threshold", 0.001))),
        clip_lo=float(tp.get("clip_lo", 0.0)),
        clip_hi=float(tp.get("clip_hi", 1.0)),
    )


def _parse_action_dynamics(raw: dict) -> ActionDynamicsConfig:
    """Parse action dynamics configuration from raw YAML dict."""
    dyn = raw.get("action_dynamics", {}) or {}
    
    return ActionDynamicsConfig(
        eh_one_time=bool(dyn.get("eh_one_time", False)),
        eh_once_min_ft=float(dyn.get("eh_once_min_ft", 3.0)),
        eh_once_max_ft=float(dyn.get("eh_once_max_ft", 5.0)),
        eh_step_ft=float(dyn.get("eh_step_ft", 1.0)),
        eh_cap_ft=float(dyn.get("eh_cap_ft", 8.0)),
        apply_RL=bool(dyn.get("apply_RL", True)),
        rl_backfill=bool(dyn.get("rl_backfill", True)),
        apply_BP=bool(dyn.get("apply_BP", True)),
        bp_backfill=bool(dyn.get("bp_backfill", False)),
        bp_sampler=str(dyn.get("bp_sampler", "lognorm_from_totals")),
    )


def _parse_files_config(raw: dict) -> FilesConfig:
    """Parse files configuration from raw YAML dict."""
    files = raw.get("files", {}) or {}
    
    return FilesConfig(
        depths_overall=files.get("depths_overall"),
        flood_events=files.get("flood_events"),
        households=files.get("households"),
    )


def load_config(
    config_path: Optional[Path] = None,
    scenario: Optional[str] = None,
    output_root: Optional[Path] = None,
    cli_overrides: Optional[dict] = None,
) -> ABMConfig:
    """Load and merge configuration from all sources.
    
    Configuration priority (highest to lowest):
    1. CLI overrides (cli_overrides dict)
    2. Environment variables
    3. Explicit function arguments
    4. YAML config file
    5. Default values
    
    Args:
        config_path: Path to YAML config file (default: config/abm_params.yaml)
        scenario: Scenario name ("baseline" or "worst")
        output_root: Custom output directory root
        cli_overrides: Dict of CLI argument overrides
        
    Returns:
        Fully populated ABMConfig object
    """
    project_root = get_project_root()
    
    # Resolve config path
    if config_path is None:
        config_path = get_default_config_path()
    config_path = Path(config_path)
    
    # Load YAML
    raw = load_yaml(config_path)
    
    # Apply CLI overrides to raw config
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                # Handle nested keys like "flood.ratio_threshold_owner"
                if "." in key:
                    section, param = key.split(".", 1)
                    raw.setdefault(section, {})[param] = value
                else:
                    raw[key] = value
    
    # Determine scenario
    yaml_scenario = str(raw.get("scenario", "baseline")).strip().lower()
    final_scenario = (scenario or yaml_scenario or "baseline").lower()
    
    # Determine output root
    env_output = os.environ.get("FLOODABM_OUTPUT_ROOT")
    final_output_root = Path(output_root or env_output or (project_root / "outputs")).resolve()
    
    # Build configuration object
    return ABMConfig(
        seed=int(raw.get("seed", 12345)),
        years_step=float(raw.get("years_step", 1.0)),
        scenario=final_scenario,
        years=raw.get("years", []),
        flood=_parse_flood_config(raw),
        tp_config=_parse_tp_config(raw),
        action_dynamics=_parse_action_dynamics(raw),
        output=OutputConfig(
            enable_plots=bool(raw.get("output", {}).get("enable_plots", True)),
            output_mode=os.environ.get("FLOODABM_OUTPUT_MODE", "full"),
        ),
        files=_parse_files_config(raw),
        project_root=project_root,
        config_path=config_path,
        output_root=final_output_root,
        _raw=raw,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def create_rng(seed: int, year_offset: int = 0) -> np.random.RandomState:
    """Create a random number generator with optional year offset.
    
    Args:
        seed: Base random seed
        year_offset: Year to add to seed (for per-year reproducibility)
        
    Returns:
        NumPy RandomState object
    """
    return np.random.RandomState(seed + year_offset)


def resolve_data_path(
    base_dir: Path,
    relative_path: Optional[str],
    fallback_dir: Optional[Path] = None
) -> Optional[Path]:
    """Resolve a data file path from config.
    
    Args:
        base_dir: Base directory for relative paths
        relative_path: Relative or absolute path from config
        fallback_dir: Alternative directory to search
        
    Returns:
        Resolved absolute path, or None if not found
    """
    if not relative_path:
        return None
    
    path = Path(relative_path)
    
    # Check if already absolute
    if path.is_absolute() and path.exists():
        return path
    
    # Try relative to base_dir
    resolved = base_dir / path
    if resolved.exists():
        return resolved
    
    # Try fallback
    if fallback_dir:
        resolved = fallback_dir / path
        if resolved.exists():
            return resolved
    
    return None
