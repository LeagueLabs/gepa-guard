"""Records of what the guard found and decided."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Regression:
    """A protected check the parent passed and the candidate failed in the minibatch.

    Parameters
    ----------
    index : int
        Position of the row in the minibatch.
    check : str
        Name of the protected objective.
    """

    index: int
    check: str


@dataclass(frozen=True)
class FloorRegression:
    """A protected check the reference passed and the candidate failed in the floor.

    Parameters
    ----------
    row_index : int
        Position of the row in the floor.
    check : str
        Name of the protected objective.
    """

    row_index: int
    check: str


@dataclass
class VetoStats:
    """Counts of the decisions a guard made.

    Parameters
    ----------
    proposals : int
        Proposals decided.
    accepted : int
        Proposals accepted.
    vetoed_unverifiable : int
        Proposals rejected because their protected checks could not be read.
    vetoed_subsample : int
        Proposals rejected for a protected regression in the minibatch.
    vetoed_floor : int
        Proposals rejected for a protected regression in the floor.
    vetoed_by_base : int
        Proposals rejected by the base criterion.
    checks : collections.Counter
        Protected regressions found, per check.
    """

    proposals: int = 0
    accepted: int = 0
    vetoed_unverifiable: int = 0
    vetoed_subsample: int = 0
    vetoed_floor: int = 0
    vetoed_by_base: int = 0
    checks: Counter[str] = field(default_factory=Counter)

    @property
    def vetoed_by_guard(self) -> int:
        """Proposals rejected for a protected regression in the minibatch or floor."""
        return self.vetoed_subsample + self.vetoed_floor

    def as_dict(self) -> dict[str, Any]:
        """Return the counts as a plain dictionary.

        Returns
        -------
        dict
            Every count, `vetoed_by_guard`, and `checks` sorted by name.
        """
        return {
            "proposals": self.proposals,
            "accepted": self.accepted,
            "vetoed_by_guard": self.vetoed_by_guard,
            "vetoed_subsample": self.vetoed_subsample,
            "vetoed_floor": self.vetoed_floor,
            "vetoed_unverifiable": self.vetoed_unverifiable,
            "vetoed_by_base": self.vetoed_by_base,
            "checks": dict(sorted(self.checks.items())),
        }
