"""Non-regression acceptance for GEPA reflective prompt optimization."""

from importlib.metadata import version

from gepa_guard.acceptance import GuardedAcceptance
from gepa_guard.floor import FloorConfig, ProtectedFloor
from gepa_guard.records import FloorRegression, Regression, VetoStats

__version__ = version("gepa-guard")

__all__ = [
    "FloorConfig",
    "FloorRegression",
    "GuardedAcceptance",
    "ProtectedFloor",
    "Regression",
    "VetoStats",
]
