"""Protected checks on rows outside the minibatch."""

import threading
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self

from gepa_guard._fingerprint import candidate_fingerprint
from gepa_guard._scores import is_pass, is_score, is_unscored, protected_names
from gepa_guard.exceptions import UnscoredCheckError
from gepa_guard.records import FloorRegression


class Evaluator(Protocol):
    """Scores rows of a floor with a candidate."""

    def __call__(
        self, rows: Sequence[Any], candidate: Mapping[str, str], /
    ) -> Sequence[Mapping[str, float | None]]:
        """Return the objective scores of each row.

        Parameters
        ----------
        rows : sequence
            Rows of the floor to score.
        candidate : mapping of str to str
            Component name to component text.

        Returns
        -------
        sequence of mapping of str to float or None
            One mapping of objective name to score per row, in the order of
            `rows`. A protected score is 0, 1 or None. None means that the row
            tests the check but has no score, for example after an error.
        """


@dataclass(frozen=True)
class FloorConfig:
    """Settings for a `ProtectedFloor`.

    Parameters
    ----------
    cache_size : int, default 128
        Number of candidates whose results are kept. 0 turns the cache off.
    batch_size : int, default 1
        Number of rows passed to one call of the evaluator.

    Raises
    ------
    ValueError
        If `cache_size` is negative or `batch_size` is below 1.
    """

    cache_size: int = 128
    batch_size: int = 1

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.cache_size < 0:
            raise ValueError("cache_size must be zero or positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


class ProtectedFloor:
    """Protected checks a reference candidate passed, checked again for each candidate.

    Calls from several threads run one at a time.

    Parameters
    ----------
    evaluate : Evaluator
        Takes a sequence of rows and a candidate. Returns one objectives mapping
        per row, in the same order.
    rows : sequence
        Rows of the floor.
    protected : iterable of str
        Objective names to protect.
    config : FloorConfig, optional
        Settings. The defaults of `FloorConfig` apply when omitted.

    Raises
    ------
    ValueError
        If `protected` is a string, holds no names, or holds an empty or
        non-string name.
    """

    def __init__(
        self,
        evaluate: Evaluator,
        rows: Sequence[Any],
        protected: Iterable[str],
        config: FloorConfig | None = None,
    ) -> None:
        self.protected = protected_names(protected)
        self.config = config if config is not None else FloorConfig()
        self._evaluate = evaluate
        self._rows = tuple(rows)
        self._required: dict[int, tuple[str, ...]] | None = None
        self._failures: Counter[int] = Counter()
        self._cache: dict[str, tuple[FloorRegression, ...]] = {}
        self._evaluations = 0
        self._lock = threading.Lock()

    @classmethod
    def from_adapter(
        cls,
        adapter: Any,
        rows: Sequence[Any],
        protected: Iterable[str],
        config: FloorConfig | None = None,
    ) -> Self:
        """Build a floor that scores rows with a gepa adapter.

        The floor calls ``adapter.evaluate(rows, candidate, capture_traces=False)``
        and reads the ``objective_scores`` of the result.

        Parameters
        ----------
        adapter : object
            A gepa adapter, or an object with the same ``evaluate`` method.
        rows : sequence
            Rows of the floor, in the form the adapter takes.
        protected : iterable of str
            Objective names to protect.
        config : FloorConfig, optional
            Settings. The defaults of `FloorConfig` apply when omitted.

        Returns
        -------
        ProtectedFloor
            An unmeasured floor.
        """
        return cls(_AdapterEvaluator(adapter), rows, protected, config)

    @property
    def measured(self) -> bool:
        """Whether `measure` has recorded a reference."""
        return self._required is not None

    @property
    def n_rows(self) -> int:
        """Rows with at least one protected check the reference passed."""
        return len(self._required or {})

    @property
    def n_cases(self) -> int:
        """Protected checks the reference passed, counted per row and check."""
        return sum(len(checks) for checks in (self._required or {}).values())

    @property
    def evaluations(self) -> int:
        """Rows scored so far."""
        return self._evaluations

    def measure(self, reference: Mapping[str, str]) -> dict[str, int]:
        """Score every row with the reference and record the checks it passed.

        Rows are scored in batches of `FloorConfig.batch_size`.

        Parameters
        ----------
        reference : mapping of str to str
            The candidate that defines the floor, normally the seed.

        Returns
        -------
        dict of str to int
            Recorded cases per protected check.

        Raises
        ------
        ValueError
            If the reference passes no row of a protected check, or if
            `evaluate` returns the wrong number of results or a protected
            score that is not 0, 1 or None. The floor then keeps its earlier state.
        """
        with self._lock:
            return self._measure(reference)

    def regressions(self, candidate: Mapping[str, str]) -> tuple[FloorRegression, ...]:
        """Return the protected checks the reference passed and the candidate fails.

        Rows are scored in batches of `FloorConfig.batch_size`. Scoring stops
        after the first batch with a regression or a missing score. Rows that
        regressed before are scored first. A result with a missing score is not
        cached.

        Parameters
        ----------
        candidate : mapping of str to str
            The candidate to check.

        Returns
        -------
        tuple of FloorRegression
            Empty when the candidate holds the floor. Otherwise the failed
            checks of the first regressed row.

        Raises
        ------
        RuntimeError
            If `measure` was not called first.
        UnscoredCheckError
            If the first such batch has no regression and has no score for a
            check that the reference passed.
        ValueError
            If `evaluate` returns the wrong number of results or a protected
            score that is not 0, 1 or None.
        """
        with self._lock:
            return self._regressions(candidate)

    def _measure(self, reference: Mapping[str, str]) -> dict[str, int]:
        """Score every row with the reference and record the checks it passed."""
        required: dict[int, tuple[str, ...]] = {}
        for batch in self._batches(range(len(self._rows))):
            for index, objectives in zip(
                batch, self._score(batch, reference), strict=True
            ):
                _check_scores(index, self.protected, objectives)
                passed = tuple(
                    name for name in self.protected if is_pass(objectives.get(name))
                )
                if passed:
                    required[index] = passed
        counts = {
            name: sum(name in checks for checks in required.values())
            for name in self.protected
        }
        missing = ", ".join(repr(name) for name, count in counts.items() if not count)
        if missing:
            raise ValueError(
                f"the reference passed no row of {missing}. "
                "Check the protected names and the evaluate function."
            )
        self._required = required
        self._failures.clear()
        self._cache.clear()
        return counts

    def _regressions(self, candidate: Mapping[str, str]) -> tuple[FloorRegression, ...]:
        """Return the regressions of a candidate, from the cache when possible."""
        if self._required is None:
            raise RuntimeError("call measure() with a reference candidate first")
        key = candidate_fingerprint(candidate)
        if key in self._cache:
            return self._cache[key]
        found = self._first_regression(self._required, candidate)
        self._remember(key, found)
        return found

    def _first_regression(
        self, required: Mapping[int, tuple[str, ...]], candidate: Mapping[str, str]
    ) -> tuple[FloorRegression, ...]:
        """Return the failed checks of the first regressed row, in priority order.

        Every regressed row of the scored batch counts as a failure. Those rows
        are scored first for the next candidate.
        """
        order = sorted(required, key=lambda row: (-self._failures[row], row))
        for batch in self._batches(order):
            regressed, unscored = [], []
            for index, objectives in zip(
                batch, self._score(batch, candidate), strict=True
            ):
                _check_scores(index, self.protected, objectives)
                if found := _lost(index, required[index], objectives):
                    regressed.append(found)
                unscored.extend(
                    UnscoredCheckError(index, name)
                    for name in required[index]
                    if is_unscored(objectives, name)
                )
            self._failures.update(found[0].row for found in regressed)
            if regressed:
                return regressed[0]
            if unscored:
                raise unscored[0]
        return ()

    def _batches(self, indices: Sequence[int]) -> Iterator[Sequence[int]]:
        """Split row indices into batches of `FloorConfig.batch_size`."""
        size = self.config.batch_size
        for start in range(0, len(indices), size):
            yield indices[start : start + size]

    def _score(
        self, indices: Sequence[int], candidate: Mapping[str, str]
    ) -> Sequence[Mapping[str, float | None]]:
        """Score a batch of rows with one call to `evaluate`."""
        self._evaluations += len(indices)
        scored = self._evaluate([self._rows[index] for index in indices], candidate)
        if len(scored) != len(indices):
            raise ValueError(
                f"evaluate returned {len(scored)} results, expected {len(indices)}"
            )
        return scored

    def _remember(self, key: str, found: tuple[FloorRegression, ...]) -> None:
        """Cache a result and evict the oldest entry when the cache is full."""
        if self.config.cache_size == 0:
            return
        if len(self._cache) >= self.config.cache_size:
            del self._cache[next(iter(self._cache))]
        self._cache[key] = found


def _lost(
    row: int, checks: Sequence[str], objectives: Mapping[str, float | None]
) -> tuple[FloorRegression, ...]:
    """Return the protected checks of a row that the candidate failed."""
    return tuple(
        FloorRegression(row=row, check=check)
        for check in checks
        if not is_pass(objectives.get(check)) and not is_unscored(objectives, check)
    )


def _check_scores(
    row: int, names: Iterable[str], objectives: Mapping[str, float | None]
) -> None:
    """Raise when a protected score of a floor row is not 0, 1 or None."""
    for name in names:
        if name in objectives and not is_score(objectives[name]):
            raise ValueError(
                f"evaluate returned a non-binary score {objectives[name]!r} "
                f"for {name!r} at floor row {row}"
            )


class _AdapterEvaluator:
    """Evaluator that scores rows with a gepa adapter."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def __call__(
        self, rows: Sequence[Any], candidate: Mapping[str, str], /
    ) -> Sequence[Mapping[str, float | None]]:
        """Return the objective scores the adapter gives each row."""
        result = self.adapter.evaluate(
            list(rows), dict(candidate), capture_traces=False
        )
        scores = getattr(result, "objective_scores", None)
        if scores is None:
            raise ValueError(
                "the adapter returned no objective_scores. "
                "It must score every protected check."
            )
        return scores
