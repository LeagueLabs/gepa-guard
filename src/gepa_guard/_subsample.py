"""Protected non-regression check on the minibatch of a proposal."""

from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from gepa_guard._scores import is_pass, is_score, is_unscored, protected_names
from gepa_guard.records import Regression


@dataclass(frozen=True)
class SubsampleFinding:
    """What the minibatch check found for one proposal.

    Parameters
    ----------
    problem : str or None
        Why the protected checks could not be read, or None when they could.
    regressions : tuple of Regression
        Protected checks the parent passed and the candidate failed. Empty
        when `problem` is set.
    checked : collections.Counter
        Protected passes of the parent compared, per check. Empty when
        `problem` is set.
    """

    problem: str | None = None
    regressions: tuple[Regression, ...] = ()
    checked: Counter[str] = field(default_factory=Counter)

    @property
    def clean(self) -> bool:
        """Whether the checks could be read and found no regression."""
        return self.problem is None and not self.regressions


class SubsampleCheck:
    """Compare a parent and its candidate on the protected objectives of a minibatch.

    A protected objective passes only at exactly 1.0. A parent pass that the
    candidate does not keep is a regression. A check that is left out or is
    None has no score. Proposals whose evaluations are missing or misaligned,
    or hold a protected score that is not 0, 1 or None, cannot be checked. A
    proposal without regressions also cannot be checked when the candidate
    has no score for a pass of its parent.

    Parameters
    ----------
    protected : iterable of str
        Objective names to protect.

    Raises
    ------
    ValueError
        If `protected` is a string, holds no names, or holds an empty or
        non-string name.
    """

    def __init__(self, protected: Iterable[str]) -> None:
        self.protected = protected_names(protected)

    def inspect(self, proposal: Any) -> SubsampleFinding:
        """Return what the check found for a proposal.

        Parameters
        ----------
        proposal : object
            A gepa ``CandidateProposal`` or an object with the same attributes.

        Returns
        -------
        SubsampleFinding
            The problem, or the regressions.
        """
        problem = _structure_problem(proposal)
        if problem is not None:
            return SubsampleFinding(problem=problem)
        before = proposal.eval_before.objective_scores
        after = proposal.eval_after.objective_scores
        problem = self._value_problem(before, after)
        if problem is not None:
            return SubsampleFinding(problem=problem)
        ids = getattr(proposal, "subsample_indices", None)
        regressions = self._regressions(before, after, ids)
        if not regressions and (problem := self._unscored(before, after)):
            return SubsampleFinding(problem=problem)
        return SubsampleFinding(
            regressions=regressions, checked=self._checked(before, after)
        )

    def _compared(
        self, before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]
    ) -> Iterator[tuple[int, str, dict[str, Any]]]:
        """Yield the row, the check and the candidate objectives of each parent pass."""
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            for name in self.protected:
                if is_pass(old.get(name)):
                    yield index, name, new

    def _checked(
        self, before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]
    ) -> Counter[str]:
        """Count the parent passes that the candidate has a score for, per check."""
        return Counter(
            name
            for _, name, new in self._compared(before, after)
            if not is_unscored(new, name)
        )

    def _unscored(
        self, before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]
    ) -> str | None:
        """Return the first parent pass that the candidate has no score for."""
        for index, name, new in self._compared(before, after):
            if is_unscored(new, name):
                return f"no score for {name!r} at minibatch row {index}"
        return None

    def _value_problem(self, before: Sequence[Any], after: Sequence[Any]) -> str | None:
        """Return the first invalid row or protected score that is not 0, 1 or None."""
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            if not isinstance(old, dict) or not isinstance(new, dict):
                return f"invalid objectives at minibatch row {index}"
            for name in self.protected:
                for row in (old, new):
                    if name in row and not is_score(row[name]):
                        return (
                            f"non-binary score {row[name]!r} for {name!r} "
                            f"at minibatch row {index}"
                        )
        return None

    def _regressions(
        self,
        before: Sequence[dict[str, Any]],
        after: Sequence[dict[str, Any]],
        ids: Sequence[Hashable] | None,
    ) -> tuple[Regression, ...]:
        """Return every protected pass the candidate lost."""
        return tuple(
            Regression(
                row=index, check=name, data_id=None if ids is None else ids[index]
            )
            for index, (old, new) in enumerate(zip(before, after, strict=True))
            for name in self.protected
            if is_pass(old.get(name))
            and not is_pass(new.get(name))
            and not is_unscored(new, name)
        )


def _structure_problem(proposal: Any) -> str | None:
    """Return why the evaluations of a proposal cannot be compared row by row."""
    before = getattr(proposal, "eval_before", None)
    after = getattr(proposal, "eval_after", None)
    if before is None or after is None:
        return "missing subsample evaluation"
    before_rows = getattr(before, "objective_scores", None)
    after_rows = getattr(after, "objective_scores", None)
    if before_rows is None or after_rows is None:
        return "missing objective_scores"
    size = len(before_rows)
    if not size or len(after_rows) != size:
        return "empty or mismatched objective_scores"
    return _alignment_problem(proposal, size)


def _alignment_problem(proposal: Any, size: int) -> str | None:
    """Return which per-row list of a proposal does not have `size` entries."""
    evaluation_scores = (
        getattr(proposal.eval_before, "scores", None),
        getattr(proposal.eval_after, "scores", None),
    )
    if any(scores is None or len(scores) != size for scores in evaluation_scores):
        return "objective_scores do not align with evaluation scores"
    subsample_scores = (
        getattr(proposal, "subsample_scores_before", None),
        getattr(proposal, "subsample_scores_after", None),
    )
    if any(scores is not None and len(scores) != size for scores in subsample_scores):
        return "objective_scores do not align with subsample scores"
    indices = getattr(proposal, "subsample_indices", None)
    if indices is not None and len(indices) != size:
        return "objective_scores do not align with subsample indices"
    return None
