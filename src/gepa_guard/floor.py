"""Protected checks on rows outside the minibatch."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from gepa_guard._fingerprint import candidate_fingerprint
from gepa_guard._scores import is_pass, protected_names
from gepa_guard.records import FloorRegression

Objectives: TypeAlias = Mapping[str, float]
Evaluate: TypeAlias = Callable[[Sequence[Any], Mapping[str, str]], Sequence[Objectives]]


@dataclass(frozen=True)
class FloorConfig:
    """Settings for a `ProtectedFloor`.

    Parameters
    ----------
    cache_size : int, default 128
        Number of candidates whose results are kept. 0 turns the cache off.

    Raises
    ------
    ValueError
        If `cache_size` is negative.
    """

    cache_size: int = 128

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.cache_size < 0:
            raise ValueError("cache_size must be zero or positive")


class ProtectedFloor:
    """Protected checks a reference candidate passed, checked again for each candidate.

    Parameters
    ----------
    evaluate : callable
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
        evaluate: Evaluate,
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
        """Rows scored so far, one call to `evaluate` each."""
        return self._evaluations

    def measure(self, reference: Mapping[str, str]) -> dict[str, int]:
        """Score every row with the reference and record the checks it passed.

        Parameters
        ----------
        reference : mapping of str to str
            The candidate that defines the floor, normally the seed.

        Returns
        -------
        dict of str to int
            Recorded cases per protected check.
        """
        required: dict[int, tuple[str, ...]] = {}
        for index, row in enumerate(self._rows):
            objectives = self._score(row, reference)
            passed = tuple(
                name for name in self.protected if is_pass(objectives.get(name))
            )
            if passed:
                required[index] = passed
        self._required = required
        self._failures.clear()
        self._cache.clear()
        return {
            name: sum(name in checks for checks in required.values())
            for name in self.protected
        }

    def regressions(self, candidate: Mapping[str, str]) -> tuple[FloorRegression, ...]:
        """Return the protected checks the reference passed and the candidate fails.

        Scoring stops at the first row with a regression. Rows that regressed
        before are scored first.

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
        """
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
        """Return the failed checks of the first regressed row, in priority order."""
        for index in sorted(required, key=lambda row: (-self._failures[row], row)):
            objectives = self._score(self._rows[index], candidate)
            found = tuple(
                FloorRegression(row_index=index, check=check)
                for check in required[index]
                if not is_pass(objectives.get(check))
            )
            if found:
                self._failures[index] += 1
                return found
        return ()

    def _score(self, row: Any, candidate: Mapping[str, str]) -> Objectives:
        """Score one row with one call to `evaluate`."""
        self._evaluations += 1
        scored = self._evaluate([row], candidate)
        if len(scored) != 1:
            raise ValueError(f"evaluate returned {len(scored)} results for 1 row")
        return scored[0]

    def _remember(self, key: str, found: tuple[FloorRegression, ...]) -> None:
        """Cache a result and evict the oldest entry when the cache is full."""
        if self.config.cache_size == 0:
            return
        if len(self._cache) >= self.config.cache_size:
            del self._cache[next(iter(self._cache))]
        self._cache[key] = found
