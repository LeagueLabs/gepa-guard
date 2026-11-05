"""Protected non-regression check on the minibatch of a proposal."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from gepa_guard._scores import is_pass, is_valid_score, protected_names
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
    """

    problem: str | None = None
    regressions: tuple[Regression, ...] = ()

    @property
    def clean(self) -> bool:
        """Whether the checks could be read and found no regression."""
        return self.problem is None and not self.regressions


class SubsampleCheck:
    """Compare a parent and its candidate on the protected objectives of a minibatch.

    A protected objective passes only at exactly 1.0. A parent pass that the
    candidate does not keep is a regression. Proposals whose evaluations are
    missing, misaligned or hold invalid protected scores cannot be checked.

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
        return SubsampleFinding(regressions=self._regressions(before, after))

    def _value_problem(self, before: Sequence[Any], after: Sequence[Any]) -> str | None:
        """Return the first invalid row or protected score."""
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            if not isinstance(old, dict) or not isinstance(new, dict):
                return f"invalid objective row at minibatch index {index}"
            for name in self.protected:
                if any(
                    name in row and not is_valid_score(row[name]) for row in (old, new)
                ):
                    return f"invalid score for {name!r} at minibatch index {index}"
        return None

    def _regressions(
        self, before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]
    ) -> tuple[Regression, ...]:
        """Return every protected pass the candidate lost."""
        return tuple(
            Regression(index=index, check=name)
            for index, (old, new) in enumerate(zip(before, after, strict=True))
            for name in self.protected
            if is_pass(old.get(name)) and not is_pass(new.get(name))
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
