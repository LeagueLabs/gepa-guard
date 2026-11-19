"""Non-regression acceptance for GEPA reflective prompt optimization."""

from importlib.metadata import version

from gepa_guard.acceptance import BaseCriterion, GuardedAcceptance
from gepa_guard.exceptions import UnguardedMergeError
from gepa_guard.floor import Evaluator, FloorConfig, ProtectedFloor
from gepa_guard.records import FloorRegression, Regression, VetoStats

__version__ = version("gepa-guard")

__all__ = [
    "BaseCriterion",
    "Evaluator",
    "FloorConfig",
    "FloorRegression",
    "GuardedAcceptance",
    "ProtectedFloor",
    "Regression",
    "UnguardedMergeError",
    "VetoStats",
]
