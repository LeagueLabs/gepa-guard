"""Acceptance criterion that rejects candidates that break a protected check."""

import threading
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from gepa_guard._subsample import SubsampleCheck
from gepa_guard.exceptions import UnguardedMergeError, UnscoredCheckError
from gepa_guard.floor import ProtectedFloor
from gepa_guard.records import FloorRegression, Regression, VetoStats


@runtime_checkable
class BaseCriterion(Protocol):
    """Criterion that decides whether gepa may accept a proposal.

    gepa's acceptance criteria and `GuardedAcceptance` match it.
    """

    def should_accept(self, proposal: Any, state: Any, /) -> bool:
        """Return whether gepa may accept a proposal.

        Parameters
        ----------
        proposal : object
            A gepa ``CandidateProposal`` or an object with the same attributes.
        state : object
            The gepa state.

        Returns
        -------
        bool
            True when the proposal is accepted.
        """


@dataclass(frozen=True)
class _Decision:
    proposal: Any
    accepted: bool
    reason: str


class GuardedAcceptance:
    """Acceptance criterion that rejects any candidate that breaks a protected check.

    The checks run in this order. The protected checks must be readable, with
    scores of 0, 1 or None. The candidate must keep every protected pass of
    its parent in the minibatch. The base criterion must accept. The candidate
    must hold the floor, when one is given. The first check that fails rejects
    the candidate. A check that is left out or is None has no score. A
    missing score for a pass of the parent or the reference, with no
    regression, makes the proposal unverifiable. After a minibatch veto, the
    guard still asks the base criterion, only to count in `stats` the vetoes
    that override it. Calls from several threads run one at a time.

    Parameters
    ----------
    base : BaseCriterion
        Criterion with ``should_accept(proposal, state)``, for example a gepa
        acceptance criterion.
    protected : iterable of str
        Objective names to protect in the minibatch.
    floor : ProtectedFloor, optional
        Measured floor that accepted candidates must hold. The guard gives it
        the minibatch scores of the candidate.

    Attributes
    ----------
    stats : VetoStats
        Counts of the decisions made so far.

    Raises
    ------
    ValueError
        If `protected` is a string, holds no names, or holds an empty or
        non-string name. Also if `floor` is not measured or protects other
        names than `protected`.
    """

    _max_pending: ClassVar[int] = 1024

    def __init__(
        self,
        base: BaseCriterion,
        protected: Iterable[str],
        floor: ProtectedFloor | None = None,
    ) -> None:
        self._subsample = SubsampleCheck(protected)
        if floor is not None:
            _check_floor(floor, self._subsample.protected)
        self.base = base
        self.floor = floor
        self.stats = VetoStats(checked=Counter(dict.fromkeys(self.protected, 0)))
        self._pending: dict[int, _Decision] = {}
        self._lock = threading.Lock()

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

        Raises
        ------
        UnguardedMergeError
            If the gepa state holds a merged candidate. gepa accepts merges
            without the guard.
        ValueError
            If the floor's evaluator returns a wrong result. See
            `ProtectedFloor.regressions`.
        """
        with self._lock:
            merged = _merged_candidate(state)
            if merged is not None:
                raise UnguardedMergeError(merged)
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
        with self._lock:
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
            reason = _unverifiable_reason(problem)
            return _Decision(proposal, False, reason)
        self.stats.checked.update(finding.checked)
        if finding.regressions:
            self.stats.vetoed_subsample += 1
            self.stats.regressed.update(
                regression.check for regression in finding.regressions
            )
            if self.base.should_accept(proposal, state):
                self.stats.vetoed_subsample_base_accepted += 1
            return _Decision(proposal, False, _minibatch_reason(finding.regressions))
        if not self.base.should_accept(proposal, state):
            self.stats.vetoed_by_base += 1
            return _Decision(proposal, False, self._base_reason(proposal, state))
        return self._decide_floor(proposal)

    def _decide_floor(self, proposal: Any) -> _Decision:
        """Check the floor for a proposal the other checks accepted."""
        try:
            regressions = (
                self.floor.regressions(
                    proposal.candidate, _minibatch_scores(proposal, self.protected)
                )
                if self.floor is not None
                else ()
            )
        except UnscoredCheckError as error:
            self.stats.vetoed_unverifiable += 1
            return _Decision(proposal, False, _unverifiable_reason(str(error)))
        if regressions:
            self.stats.vetoed_floor += 1
            self.stats.regressed.update(regression.check for regression in regressions)
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


def _check_floor(floor: ProtectedFloor, protected: tuple[str, ...]) -> None:
    """Raise when a floor cannot guard the given protected names."""
    if not floor.measured:
        raise ValueError("floor must be measured before it can guard")
    if set(floor.protected) != set(protected):
        raise ValueError(
            f"the floor protects {floor.protected} but the guard protects "
            f"{protected}. Use the same names."
        )


def _minibatch_scores(
    proposal: Any, protected: Sequence[str]
) -> dict[Hashable, dict[str, Any]]:
    """Return the protected minibatch scores of the candidate by gepa data id.

    A data id whose copies in the minibatch have different scores is left out.
    """
    ids = getattr(proposal, "subsample_indices", None)
    if ids is None:
        return {}
    scores: dict[Hashable, dict[str, Any]] = {}
    conflicts: set[Hashable] = set()
    for data_id, objectives in zip(
        ids, proposal.eval_after.objective_scores, strict=True
    ):
        view = {name: objectives[name] for name in protected if name in objectives}
        if scores.setdefault(data_id, view) != view:
            conflicts.add(data_id)
    return {key: view for key, view in scores.items() if key not in conflicts}


def _merged_candidate(state: Any) -> int | None:
    """Return the index of the first candidate with more than one parent."""
    lineage = getattr(state, "parent_program_for_candidate", None) or ()
    return next(
        (index for index, parents in enumerate(lineage) if len(parents) > 1), None
    )


def _minibatch_reason(regressions: Sequence[Regression]) -> str:
    """Describe protected regressions in the minibatch."""
    cases = ", ".join(
        f"{regression.check} at row {regression.row}{_data_id_note(regression)}"
        for regression in regressions
    )
    return f"rejected: protected regression in the minibatch on {cases}"


def _data_id_note(regression: Regression) -> str:
    """Describe the dataset id of a regression, when it is known."""
    return "" if regression.data_id is None else f" (data id {regression.data_id})"


def _unverifiable_reason(problem: str) -> str:
    """Describe why the protected checks of a proposal could not be verified."""
    return f"rejected: protected checks could not be verified ({problem})"


def _floor_reason(regressions: Sequence[FloorRegression]) -> str:
    """Describe protected regressions in the floor."""
    cases = ", ".join(
        f"{regression.check} at row {regression.row}" for regression in regressions
    )
    return f"rejected: protected regression in the floor on {cases}"
