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

from gepa_guard import (
    FloorConfig,
    GuardedAcceptance,
    ProtectedFloor,
    UnguardedMergeError,
)

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


class TestAdapterFloor:
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_keeps_emergency(
        self,
        optimize: Optimize,
        adapter: Any,
        protected_rows: list[Any],
        unprotected_rows: list[Any],
        batch_size: int,
    ) -> None:
        config = FloorConfig(batch_size=batch_size)
        floor = ProtectedFloor.from_adapter(adapter, protected_rows, PROTECTED, config)
        floor.measure({"prompt": "emergency"})
        assert keeps_emergency(optimize(guard(floor), unprotected_rows))


class TestReusedFloorScores:
    def run(
        self,
        optimize: Optimize,
        adapter: Any,
        all_rows: list[Any],
        protected_rows: list[Any],
        ids: list[int] | None,
    ) -> tuple[dict[str, Any], list[Any], int]:
        floor = ProtectedFloor.from_adapter(adapter, protected_rows, PROTECTED, ids=ids)
        floor.measure({"prompt": "emergency"})
        criterion = guard(floor)
        result = optimize(criterion, all_rows)
        return criterion.stats.as_dict(), result.candidates, floor.evaluations

    def test_same_decisions_with_fewer_evaluations(
        self,
        optimize: Optimize,
        adapter: Any,
        all_rows: list[Any],
        protected_rows: list[Any],
    ) -> None:
        scored = self.run(optimize, adapter, all_rows, protected_rows, ids=None)
        reused = self.run(optimize, adapter, all_rows, protected_rows, ids=[0])
        assert reused[:2] == scored[:2]
        assert reused[2] < scored[2]

    def test_measures_from_known_scores(
        self,
        optimize: Optimize,
        adapter: Any,
        protected_rows: list[Any],
        unprotected_rows: list[Any],
    ) -> None:
        floor = ProtectedFloor.from_adapter(adapter, protected_rows, PROTECTED)
        floor.measure_scores(
            adapter.objectives(protected_rows, {"prompt": "emergency"})
        )
        assert keeps_emergency(optimize(guard(floor), unprotected_rows))


class TestUngradedRows:
    def test_runs_with_an_ungraded_valset_row(
        self,
        optimize: Optimize,
        all_rows: list[Any],
        ungraded_valset: list[Any],
        ungraded_adapter: Any,
    ) -> None:
        result = optimize(guard(), all_rows, ungraded_valset, ungraded_adapter)
        assert keeps_emergency(result)
        assert len(result.candidates) > 1

    def test_ungraded_child_is_unverifiable(
        self, optimize: Optimize, all_rows: list[Any], ungraded_children_adapter: Any
    ) -> None:
        criterion = guard()
        result = optimize(criterion, all_rows, scorer=ungraded_children_adapter)
        assert result.candidates == [{"prompt": "emergency"}]
        stats = criterion.stats
        assert stats.vetoed_unverifiable == stats.proposals > 0


class TestMerges:
    def test_unguarded_run_accepts_a_merge(
        self, optimize_with_merges: Optimize
    ) -> None:
        result = optimize_with_merges(StrictImprovementAcceptance())
        assert any(len(parents) > 1 for parents in result.parents)

    def test_guarded_run_stops_at_the_merge(
        self, optimize_with_merges: Optimize
    ) -> None:
        with pytest.raises(UnguardedMergeError, match="use_merge=False"):
            optimize_with_merges(guard())
