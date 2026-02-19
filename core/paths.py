# -*- coding: utf-8 -*-
"""
Path Management for FLOODABM
============================

Centralized path resolution and directory creation for simulation outputs.
This module ensures consistent directory structure across all entry points.

Output Structure:
    outputs/
    ├── baseline/           # Scenario outputs
    │   ├── decisions/
    │   ├── finance/
    │   ├── states/
    │   ├── visualization/
    │   └── vulnerability/
    ├── worst/              # Worst-case scenario
    ├── experiments/        # Sensitivity analysis
    └── montecarlo/         # Monte Carlo outputs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OutputPaths:
    """Container for all output directory paths.
    
    Automatically creates directories on initialization.
    
    Attributes:
        root: Base output directory (e.g., outputs/baseline/)
        decisions: Agent decision CSVs
        finance: Insurance/finance results
        states: State snapshots per year
        visualization: Plots and figures
        vulnerability: Flood damage outputs
    """
    root: Path
    decisions: Path = field(init=False)
    finance: Path = field(init=False)
    states: Path = field(init=False)
    visualization: Path = field(init=False)
    vulnerability: Path = field(init=False)
    flood_damage: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize subdirectory paths and create them."""
        self.root = Path(self.root)
        self.decisions = self.root / "decisions"
        self.finance = self.root / "finance"
        self.states = self.root / "states"
        self.visualization = self.root / "visualization"
        self.vulnerability = self.root / "vulnerability"
        self.flood_damage = self.vulnerability / "flood_damage"
        
        # Create all directories
        self.create_all()
    
    def create_all(self) -> None:
        """Create all output directories if they don't exist."""
        for path in [
            self.root,
            self.decisions,
            self.finance,
            self.states,
            self.visualization,
            self.vulnerability,
            self.flood_damage,
        ]:
            path.mkdir(parents=True, exist_ok=True)
    
    def year_file(self, subdir: str, prefix: str, year: int, ext: str = "csv") -> Path:
        """Generate a year-specific output file path.
        
        Args:
            subdir: Subdirectory name (e.g., "decisions", "finance")
            prefix: File prefix (e.g., "decisions_mgmix")
            year: Year number
            ext: File extension without dot (default: "csv")
            
        Returns:
            Full path like outputs/baseline/decisions/decisions_mgmix_2011.csv
        """
        directory = getattr(self, subdir, self.root / subdir)
        return directory / f"{prefix}_{year}.{ext}"


def create_scenario_paths(
    output_root: Path,
    scenario: str = "baseline",
) -> OutputPaths:
    """Create output paths for a scenario simulation.
    
    Args:
        output_root: Base output directory (e.g., FLOODABM/outputs)
        scenario: Scenario name ("baseline" or "worst")
        
    Returns:
        OutputPaths object with all directories created
    """
    scenario_dir = output_root / scenario.lower()
    return OutputPaths(root=scenario_dir)


def create_experiment_paths(
    output_root: Path,
    experiment_name: str,
) -> OutputPaths:
    """Create output paths for sensitivity analysis experiments.
    
    Args:
        output_root: Base output directory
        experiment_name: Experiment identifier (e.g., "MG=0.3_NMG=0.4")
        
    Returns:
        OutputPaths object for the experiment
    """
    exp_dir = output_root / "experiments" / experiment_name
    return OutputPaths(root=exp_dir)


def create_montecarlo_paths(output_root: Path) -> dict[str, Path]:
    """Create output paths for Monte Carlo analysis.
    
    Args:
        output_root: Base output directory
        
    Returns:
        Dictionary with 'root' and 'fig' paths
    """
    mc_root = output_root / "montecarlo"
    fig_dir = mc_root / "fig"
    
    mc_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    return {"root": mc_root, "fig": fig_dir}


def get_project_root() -> Path:
    """Get the FLOODABM project root directory."""
    return Path(__file__).resolve().parent.parent


def resolve_file_path(
    base_dir: Path,
    relative_path: Optional[str],
    required: bool = False,
) -> Optional[Path]:
    """Resolve a file path from configuration.
    
    Args:
        base_dir: Base directory for relative paths
        relative_path: Relative or absolute path string from config
        required: If True, raise error when file not found
        
    Returns:
        Resolved absolute path, or None if not found
        
    Raises:
        FileNotFoundError: If required=True and file doesn't exist
    """
    if not relative_path:
        if required:
            raise FileNotFoundError("Required file path not specified")
        return None
    
    path = Path(relative_path)
    
    # Check if already absolute
    if path.is_absolute():
        if path.exists():
            return path
        if required:
            raise FileNotFoundError(f"File not found: {path}")
        return None
    
    # Try relative to base_dir
    resolved = base_dir / path
    if resolved.exists():
        return resolved
    
    # Try relative to project root
    project_root = get_project_root()
    resolved = project_root / path
    if resolved.exists():
        return resolved
    
    if required:
        raise FileNotFoundError(f"File not found: {relative_path} (searched in {base_dir})")
    return None
