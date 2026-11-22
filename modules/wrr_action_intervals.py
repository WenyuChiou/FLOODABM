"""Saved one-at-a-time draw-bound settings from revised WRR Figure S4."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntervalCase:
    key: str
    bounds: tuple[float, float]


CONTROL_BOUNDS = {
    "FI": (0.35, 0.55),
    "FI_renter": (0.70, 0.90),
    "EH": (0.30, 0.60),
    "BP": (0.25, 0.65),
    "RL": (0.30, 0.95),
}
# Variation order is lower, higher, narrower, then wider, as in the published design.
_VARIANTS = {
    "FI_homeowner_outside_SFHA": ("FI", ((0.30, 0.50), (0.40, 0.60), (0.40, 0.50), (0.30, 0.60))),
    "FI_renter": ("FI_renter", ((0.65, 0.85), (0.75, 0.95), (0.75, 0.85), (0.65, 0.95))),
    "EH": ("EH", ((0.25, 0.55), (0.35, 0.65), (0.35, 0.55), (0.25, 0.65))),
    "BP": ("BP", ((0.20, 0.60), (0.30, 0.70), (0.30, 0.60), (0.20, 0.70))),
    "RL": ("RL", ((0.25, 0.90), (0.35, 1.00), (0.35, 0.90), (0.25, 1.00))),
}
CASES = {
    f"EX01_{component}_{variation}": IntervalCase(key, bounds)
    for component, (key, intervals) in _VARIANTS.items()
    for variation, bounds in zip(("lower", "higher", "narrower", "wider"), intervals, strict=True)
}


def apply_wrr_action_interval(config: dict, case_id: str) -> None:
    """Change only the recorded draw-bound key after checking the control intervals."""
    sfha = config.get("sfha_initialization") or {}
    current = config.get("draw_bounds") or {}
    if not sfha.get("enabled") or tuple(sfha.get("fi_draw_bounds_inside") or ()) != (0.0, 0.1):
        raise ValueError("Figure S4 requires the revised SFHA initialization and inside-SFHA FI interval")
    if any(tuple(current.get(key) or ()) != expected for key, expected in CONTROL_BOUNDS.items()):
        raise ValueError("Figure S4 requires the revised control action intervals")
    case = CASES[case_id]
    current[case.key] = list(case.bounds)
