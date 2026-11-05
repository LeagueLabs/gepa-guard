"""Acceptance criterion that rejects candidates that break a protected check."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from gepa_guard._subsample import SubsampleCheck
from gepa_guard.floor import ProtectedFloor
from gepa_guard.records import FloorRegression, Regression, VetoStats


class _Criterion(Protocol):
    def should_accept(self, proposal: Any, state: Any) -> bool: ...


@dataclass(frozen=True)
class _Decision:
    proposal: Any
    accepted: bool
    reason: str


class GuardedAcceptance:
    """Acceptance criterion that rejects any candidate that breaks a protected check.

    The checks run in this order. The protected checks must be readable. The
    candidate must keep every protected pass of its parent in the minibatch.
    The base criterion must accept. The candidate must hold the floor, when
    one is given. The first check that fails rejects the candidate.

    Parameters
    ----------
    base : object
        Criterion with ``should_accept(proposal, state)``, for example a gepa
        acceptance criterion.
    protected : iterable of str
        Objective names to protect in the minibatch.
    floor : ProtectedFloor, optional
        Measured floor that accepted candidates must hold.

    Attributes
    ----------
    stats : VetoStats
        Counts of the decisions made so far.

    Raises
    ------
    ValueError
        If `protected` is a string, holds no names, or holds an empty or
        non-string name. Also if `floor` is not measured.
    """

    _max_pending: ClassVar[int] = 1024

    def __init__(
        self,
        base: _Criterion,
        protected: Iterable[str],
        floor: ProtectedFloor | None = None,
    ) -> None:
        if floor is not None and not floor.measured:
            raise ValueError("floor must be measured before it can guard")
        self.base = base
        self.floor = floor
        self.stats = VetoStats()
        self._subsample = SubsampleCheck(protected)
        self._pending: dict[int, _Decision] = {}

    @property
    def protected(self) -> tuple[str, ...]:
        """Objective names protected in the minibatch."""
        return self._subsample.protected

    def should_accept(self, proposal: Any, state: Any) -> bool:
        """Decide whether gepa may accept a proposal.

        Parameters
        ----------
        proposal : object
            A gepa ``CandidateProposal`` or an object with the same attributes.
        state : object
            The gepa state. It is passed to the base criterion unchanged.

        Returns
        -------
        bool
            True when every check passes.
        """
        decision = self._decide(proposal, state)
        self._pending.pop(id(proposal), None)
        if not decision.accepted:
            self._hold(decision)
        return decision.accepted

    def reject_reason(self, proposal: Any, state: Any) -> str:
        """Return why `should_accept` rejected a proposal, without deciding again.

        Each reason is returned once. The guard keeps the most recent
        unexplained rejections.

        Parameters
        ----------
        proposal : object
            A proposal that `should_accept` rejected.
        state : object
            The gepa state. It is not used.

        Returns
        -------
        str
            The reason recorded by `should_accept`.
        """
        decision = self._pending.pop(id(proposal), None)
        if decision is None:
            return "no rejection was recorded for this proposal"
        return decision.reason

    def _hold(self, decision: _Decision) -> None:
        """Keep a rejection until `reject_reason` asks for it."""
        if len(self._pending) >= self._max_pending:
            del self._pending[next(iter(self._pending))]
        self._pending[id(decision.proposal)] = decision

    def _decide(self, proposal: Any, state: Any) -> _Decision:
        """Run the checks in order and record the outcome in `stats`."""
        self.stats.proposals += 1
        finding = self._subsample.inspect(proposal)
        problem = finding.problem or self._floor_problem(proposal)
        if problem is not None:
            self.stats.vetoed_unverifiable += 1
            reason = f"rejected: protected checks could not be verified ({problem})"
            return _Decision(proposal, False, reason)
        if finding.regressions:
            self.stats.vetoed_subsample += 1
            self.stats.checks.update(
                regression.check for regression in finding.regressions
            )
            return _Decision(proposal, False, _minibatch_reason(finding.regressions))
        if not self.base.should_accept(proposal, state):
            self.stats.vetoed_by_base += 1
            return _Decision(proposal, False, self._base_reason(proposal, state))
        return self._decide_floor(proposal)

    def _decide_floor(self, proposal: Any) -> _Decision:
        """Check the floor for a proposal the other checks accepted."""
        regressions = (
            self.floor.regressions(proposal.candidate) if self.floor is not None else ()
        )
        if regressions:
            self.stats.vetoed_floor += 1
            self.stats.checks.update(regression.check for regression in regressions)
            return _Decision(proposal, False, _floor_reason(regressions))
        self.stats.accepted += 1
        return _Decision(proposal, True, "accepted")

    def _floor_problem(self, proposal: Any) -> str | None:
        """Return why the floor cannot check a proposal."""
        if self.floor is None or getattr(proposal, "candidate", None):
            return None
        return "the proposal has no candidate to check against the floor"

    def _base_reason(self, proposal: Any, state: Any) -> str:
        """Return the explanation of the base criterion, or name it."""
        explain = getattr(self.base, "reject_reason", None)
        if callable(explain):
            return str(explain(proposal, state))
        return (
            f"rejected: the base criterion ({type(self.base).__name__}) did not accept"
        )


def _minibatch_reason(regressions: Sequence[Regression]) -> str:
    """Describe protected regressions in the minibatch."""
    cases = ", ".join(
        f"{regression.check} at index {regression.index}" for regression in regressions
    )
    return f"rejected: protected regression in the minibatch on {cases}"


def _floor_reason(regressions: Sequence[FloorRegression]) -> str:
    """Describe protected regressions in the floor."""
    cases = ", ".join(
        f"{regression.check} at row {regression.row_index}"
        for regression in regressions
    )
    return f"rejected: protected regression in the floor on {cases}"
