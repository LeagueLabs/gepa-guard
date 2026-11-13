from collections.abc import Callable
from typing import Any

import pytest
from gepa.strategies.acceptance import StrictImprovementAcceptance
from gepa.strategies.proposal_sampling import IndependentSampling
from gepa.strategies.proposal_selection import (
    AllImprovements,
    BestImprovement,
    TopKImprovements,
)

from gepa_guard import GuardedAcceptance, ProtectedFloor

PROTECTED = ("emergency",)
LOSES_EMERGENCY = {"prompt": "alpha beta"}
SELECTIONS = (AllImprovements(), BestImprovement(), TopKImprovements(2))

STRATEGIES = [
    pytest.param({}, id="defaults"),
    pytest.param({"sampling_strategy": IndependentSampling(2)}, id="independent-2"),
    *(
        pytest.param(
            {"sampling_strategy": IndependentSampling(2), "selection_strategy": s},
            id=f"independent-2-{type(s).__name__}",
        )
        for s in SELECTIONS
    ),
]

Optimize = Callable[..., Any]


def keeps_emergency(result: Any) -> bool:
    return all("emergency" in candidate["prompt"] for candidate in result.candidates)


def guard(floor: ProtectedFloor | None = None) -> GuardedAcceptance:
    return GuardedAcceptance(StrictImprovementAcceptance(), PROTECTED, floor=floor)


@pytest.fixture
def floor(adapter: Any, protected_rows: list[Any]) -> ProtectedFloor:
    floor = ProtectedFloor(adapter.objectives, protected_rows, PROTECTED)
    floor.measure({"prompt": "emergency"})
    return floor


class TestMinibatchGuard:
    def test_unguarded_run_loses_emergency(
        self, optimize: Optimize, all_rows: list[Any]
    ) -> None:
        result = optimize(StrictImprovementAcceptance(), all_rows)
        assert LOSES_EMERGENCY in result.candidates

    @pytest.mark.parametrize("options", STRATEGIES)
    def test_guarded_run_keeps_emergency(
        self, optimize: Optimize, all_rows: list[Any], options: dict[str, Any]
    ) -> None:
        criterion = guard()
        result = optimize(criterion, all_rows, **options)
        assert keeps_emergency(result)
        assert criterion.stats.vetoed_subsample >= 1

    def test_logs_the_reason_in_a_batch(
        self, optimize: Optimize, all_rows: list[Any], logger: Any
    ) -> None:
        optimize(guard(), all_rows, sampling_strategy=IndependentSampling(2))
        expected = "rejected: protected regression in the minibatch on emergency"
        assert any(expected in line for line in logger.lines)


class TestFloorGuard:
    def test_minibatch_guard_misses_unsampled_row(
        self, optimize: Optimize, unprotected_rows: list[Any]
    ) -> None:
        result = optimize(guard(), unprotected_rows)
        assert LOSES_EMERGENCY in result.candidates

    @pytest.mark.parametrize("options", STRATEGIES)
    def test_floor_keeps_emergency(
        self,
        optimize: Optimize,
        unprotected_rows: list[Any],
        floor: ProtectedFloor,
        options: dict[str, Any],
    ) -> None:
        criterion = guard(floor)
        result = optimize(criterion, unprotected_rows, **options)
        assert keeps_emergency(result)
        assert criterion.stats.vetoed_floor >= 1


class TestMerges:
    def test_unguarded_run_accepts_a_merge(
        self, optimize_with_merges: Optimize
    ) -> None:
        result = optimize_with_merges(StrictImprovementAcceptance())
        assert any(len(parents) > 1 for parents in result.parents)

    def test_guarded_run_stops_at_the_merge(
        self, optimize_with_merges: Optimize
    ) -> None:
        with pytest.raises(RuntimeError, match="use_merge=False"):
            optimize_with_merges(guard())
