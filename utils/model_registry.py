# -*- coding: utf-8 -*-
"""
Model Registry for FLOODABM
===========================

Auto-detect and manage model configurations.
Reads from centralized config file: config/models/models_config.yaml
Falls back to auto-detection if config file is missing.

Models are stored in 'models/' folder at project root:
  models/
  ├── baseline/
  ├── baseline_fast/
  ├── optimized/
  └── optimized_fast/

Example:
    >>> from utils.model_registry import ModelRegistry
    >>> registry = ModelRegistry(Path("."))  # project root
    >>> registry.print_summary()
    Found 2 model(s):
      • baseline: Baseline Model (8 pkl) ✓ calibrators ✓ cached
      • optimized: Optimized Model (8 pkl) ✓ calibrators ✓ cached
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class ModelRegistry:
    """Registry for auto-detecting and managing ABM models."""
    
    def __init__(self, project_root: Path = None):
        """
        Initialize registry.
        
        Args:
            project_root: Path to project root directory (where main.py is)
        """
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.models_dir = self.project_root / "models"
        self._models: Dict[str, dict] = {}
        self._load()
    
    def _load(self):
        """Load model configurations from centralized config AND auto-detect others."""
        config_file = self.config_dir / "models" / "models_config.yaml"
        
        if config_file.exists() and HAS_YAML:
            self._load_from_config(config_file)
        
        # Always scan directories to find new models not in config
        self._scan_directories()
    
    def _load_from_config(self, config_file: Path):
        """Load models from centralized config file."""
        try:
            config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
            models_config = config.get("models", {})
            
            for model_key, model_info in models_config.items():
                if not isinstance(model_info, dict):
                    continue
                
                # Resolve model path (relative to project root)
                model_path = model_info.get("path", f"models/{model_key}")
                if not Path(model_path).is_absolute():
                    model_path = self.project_root / model_path
                model_path = Path(model_path)
                
                if not model_path.exists():
                    continue
                
                # Count pkl files
                pkl_files = list(model_path.glob("*_*.pkl"))
                
                # Detect cache directory (sibling *_fast folder)
                cache_dir = None
                if (model_path / "cache").exists():
                    cache_dir = model_path / "cache"
                elif Path(str(model_path) + "_fast").exists():
                    cache_dir = Path(str(model_path) + "_fast")
                
                # Also check for npz files (either in same dir or cache dir)
                npz_files = list(model_path.glob("*_*.npz"))
                cache_npz_files = list(cache_dir.glob("*_*.npz")) if cache_dir else []
                
                # Skip if no pkl AND no npz files found
                if not pkl_files and not npz_files and not cache_npz_files:
                    continue
                
                # Count available weight files
                weight_count = len(pkl_files) or len(npz_files) or len(cache_npz_files)
                
                self._models[model_key] = {
                    "path": model_path,
                    "name": model_info.get("name", model_key),
                    "description": model_info.get("description", ""),
                    "version": model_info.get("version", "1.0"),
                    "meta": model_info.get("meta", {}),
                    "overrides": model_info.get("overrides", {}),
                    "pkl_count": len(pkl_files),
                    "npz_count": len(npz_files) + len(cache_npz_files),
                    "has_calibrators": (model_path / "best_calibrators.json").exists(),
                    "cache_dir": cache_dir,
                }
        except Exception as e:
            print(f"[warn] Failed to load config: {e}, falling back to auto-detect")
            # If config load fails significantly, we still want to scan
            pass
    
    def _scan_directories(self):
        """Scan for model directories inside models/ folder."""
        if not self.models_dir.exists():
            return
        for d in self.models_dir.iterdir():
            # Skip cache directories (*_fast)
            if d.is_dir() and not d.name.endswith("_fast"):
                # If model is already registered (e.g. from config), skip it
                if d.name in self._models:
                    continue
                self._register_from_dir(d)
    
    def _register_from_dir(self, model_dir: Path):
        """Register a model from its directory (fallback mode)."""
        # Check for pkl files (required)
        pkl_files = list(model_dir.glob("*_*.pkl"))
        if not pkl_files:
            return
        
        # Load VERSION_INFO.json if exists
        config = {}
        ver_path = model_dir / "VERSION_INFO.json"
        if ver_path.exists():
            try:
                ver = json.loads(ver_path.read_text(encoding="utf-8"))
                config = {
                    "name": ver.get("version", model_dir.name),
                    "description": ver.get("description", ""),
                    "meta": ver.get("improvements", {}),
                }
            except Exception:
                pass
        
        # Detect cache directory
        cache_dir = None
        if (model_dir / "cache").exists():
            cache_dir = model_dir / "cache"
        elif Path(str(model_dir) + "_fast").exists():
            cache_dir = Path(str(model_dir) + "_fast")
        
        self._models[model_dir.name] = {
            "path": model_dir,
            "name": config.get("name", model_dir.name),
            "description": config.get("description", ""),
            "version": config.get("version", "1.0"),
            "meta": config.get("meta", {}),
            "overrides": config.get("overrides", {}),
            "pkl_count": len(pkl_files),
            "has_calibrators": (model_dir / "best_calibrators.json").exists(),
            "cache_dir": cache_dir,
        }
    
    def list_models(self) -> List[str]:
        """Return sorted list of model directory names."""
        return sorted(self._models.keys())
    
    def get_model(self, name: str) -> Optional[dict]:
        """Get model info by directory name."""
        return self._models.get(name)
    
    def get_all(self) -> Dict[str, dict]:
        """Get all registered models."""
        return self._models.copy()
    
    def get_path(self, name: str) -> Optional[Path]:
        """Get model directory path by name."""
        model = self._models.get(name)
        return model["path"] if model else None
    
    def print_summary(self):
        """Print human-readable model summary."""
        print(f"Found {len(self._models)} model(s):")
        for name in self.list_models():
            m = self._models[name]
            flags = []
            if m["has_calibrators"]:
                flags.append("[OK] calibrators")
            if m["cache_dir"]:
                flags.append("[OK] cached")
            flag_str = " ".join(flags)
            # Show pkl or npz count (prefer npz if no pkl)
            file_count = m['pkl_count'] if m['pkl_count'] > 0 else m.get('npz_count', 0)
            file_type = "pkl" if m['pkl_count'] > 0 else "npz"
            print(f"  * {name}: {m['name']} ({file_count} {file_type}) {flag_str}")
            if m["description"]:
                desc = m["description"][:60] + "..." if len(m["description"]) > 60 else m["description"]
                print(f"      {desc}")
