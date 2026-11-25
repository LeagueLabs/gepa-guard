"""Records of what the guard found and decided."""

from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Regression:
    """A protected check the parent passed and the candidate failed in the minibatch.

    Parameters
    ----------
    row : int
        Position of the row in the minibatch.
    check : str
        Name of the protected objective.
    data_id : hashable, optional
        gepa id of the row in the dataset, when the proposal carries it.
    """

    row: int
    check: str
    data_id: Hashable | None = None


@dataclass(frozen=True)
class FloorRegression:
    """A protected check the reference passed and the candidate failed in the floor.

    Parameters
    ----------
    row : int
        Position of the row in the floor.
    check : str
        Name of the protected objective.
    """

    row: int
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
        Proposals rejected because their protected checks could not be read,
        or had no score for a pass of the parent or the reference.
    vetoed_subsample : int
        Proposals rejected for a protected regression in the minibatch.
    vetoed_floor : int
        Proposals rejected for a protected regression in the floor.
    vetoed_by_base : int
        Proposals rejected by the base criterion.
    vetoed_subsample_base_accepted : int
        Minibatch vetoes of proposals that the base criterion accepts.
    checked : collections.Counter
        Protected passes of the parent compared in the minibatch, per check.
    regressed : collections.Counter
        Protected regressions found in the minibatch or the floor, per check.
    """

    proposals: int = 0
    accepted: int = 0
    vetoed_unverifiable: int = 0
    vetoed_subsample: int = 0
    vetoed_floor: int = 0
    vetoed_by_base: int = 0
    vetoed_subsample_base_accepted: int = 0
    checked: Counter[str] = field(default_factory=Counter)
    regressed: Counter[str] = field(default_factory=Counter)

    @property
    def vetoed_by_guard(self) -> int:
        """Proposals rejected for a protected regression in the minibatch or floor."""
        return self.vetoed_subsample + self.vetoed_floor

    @property
    def overrode_base(self) -> int:
        """Vetoes of proposals that the base criterion accepts.

        Every floor veto counts, because the floor runs only after the base
        criterion accepts.
        """
        return self.vetoed_subsample_base_accepted + self.vetoed_floor

    def as_dict(self) -> dict[str, Any]:
        """Return the counts as a plain dictionary.

        Returns
        -------
        dict
            Every count, `vetoed_by_guard` and `overrode_base`. `checked` and
            `regressed` are sorted by check name.
        """
        return {
            "proposals": self.proposals,
            "accepted": self.accepted,
            "vetoed_by_guard": self.vetoed_by_guard,
            "vetoed_subsample": self.vetoed_subsample,
            "vetoed_floor": self.vetoed_floor,
            "vetoed_unverifiable": self.vetoed_unverifiable,
            "vetoed_by_base": self.vetoed_by_base,
            "vetoed_subsample_base_accepted": self.vetoed_subsample_base_accepted,
            "overrode_base": self.overrode_base,
            "checked": dict(sorted(self.checked.items())),
            "regressed": dict(sorted(self.regressed.items())),
        }
