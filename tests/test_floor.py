from typing import Any

import pytest

from gepa_guard import (
    FloorConfig,
    FloorRegression,
    ProtectedFloor,
    UnscoredCheckError,
)


@pytest.fixture
def floor(
    model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
) -> ProtectedFloor:
    floor = ProtectedFloor(model, rows, protected)
    floor.measure({"prompt": "seed"})
    return floor


@pytest.fixture
def batched(
    model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
) -> ProtectedFloor:
    floor = ProtectedFloor(model, rows, protected, FloorConfig(batch_size=2))
    floor.measure(candidate("seed"))
    return floor


def candidate(text: str) -> dict[str, str]:
    return {"prompt": text}


class TestFloorConfig:
    def test_default_cache_size(self) -> None:
        assert FloorConfig().cache_size == 128

    def test_default_batch_size(self) -> None:
        assert FloorConfig().batch_size == 1

    def test_rejects_batch_size_below_one(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            FloorConfig(batch_size=0)

    def test_rejects_negative_cache_size(self) -> None:
        with pytest.raises(ValueError, match="cache_size"):
            FloorConfig(cache_size=-1)


class TestConstruction:
    def test_rejects_empty_protected(
        self, model: Any, rows: list[dict[str, str]]
    ) -> None:
        with pytest.raises(ValueError, match="protected"):
            ProtectedFloor(model, rows, ())

    def test_starts_unmeasured(
        self, model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        assert not ProtectedFloor(model, rows, protected).measured

    def test_check_before_measure_raises(
        self, model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        with pytest.raises(RuntimeError, match="measure"):
            ProtectedFloor(model, rows, protected).regressions(candidate("holds"))


class TestMeasure:
    def test_counts_cases_per_check(
        self, model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        floor = ProtectedFloor(model, rows, protected)
        assert floor.measure(candidate("seed")) == {"emergency": 2, "encoding": 1}

    def test_drops_rows_the_reference_fails(
        self, model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        floor = ProtectedFloor(model, rows, protected)
        floor.measure(candidate("fails-e2"))
        assert (floor.n_rows, floor.n_cases) == (2, 2)

    @pytest.mark.parametrize("names", [("emergency", "emergncy"), ("emergncy",)])
    def test_rejects_a_check_without_cases(
        self, model: Any, rows: list[dict[str, str]], names: tuple[str, ...]
    ) -> None:
        floor = ProtectedFloor(model, rows, names)
        with pytest.raises(ValueError, match="'emergncy'"):
            floor.measure(candidate("seed"))

    def test_failed_measure_leaves_floor_unmeasured(
        self, model: Any, rows: list[dict[str, str]]
    ) -> None:
        floor = ProtectedFloor(model, rows, ("emergncy",))
        with pytest.raises(ValueError, match="no row"):
            floor.measure(candidate("seed"))
        assert not floor.measured

    def test_rejects_a_non_binary_reference_score(
        self, model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        floor = ProtectedFloor(model, rows, protected)
        with pytest.raises(ValueError, match=r"non-binary score 0\.5 for 'emergency'"):
            floor.measure(candidate("partial-e1"))

    def test_scores_each_row_once(self, floor: ProtectedFloor, model: Any) -> None:
        assert model.scored == ["e1", "e2", "x1"]

    def test_clears_the_cache(self, floor: ProtectedFloor, model: Any) -> None:
        floor.regressions(candidate("holds"))
        floor.measure(candidate("seed"))
        floor.regressions(candidate("holds"))
        assert floor.evaluations == 3 + 3 + 3 + 3


class TestRegressions:
    def test_rejects_a_non_binary_candidate_score(self, floor: ProtectedFloor) -> None:
        with pytest.raises(ValueError, match="at floor row 0"):
            floor.regressions(candidate("partial-e1"))

    def test_rejects_a_non_binary_score_of_any_protected_check(
        self, floor: ProtectedFloor
    ) -> None:
        with pytest.raises(ValueError, match="for 'encoding' at floor row 0"):
            floor.regressions(candidate("noisy-e1"))

    def test_holding_candidate_has_none(self, floor: ProtectedFloor) -> None:
        assert floor.regressions(candidate("holds")) == ()

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("breaks-e2", FloorRegression(1, "emergency")),
            ("breaks-x1", FloorRegression(2, "encoding")),
        ],
    )
    def test_reports_the_regressed_case(
        self, floor: ProtectedFloor, text: str, expected: FloorRegression
    ) -> None:
        assert floor.regressions(candidate(text)) == (expected,)

    def test_stops_at_first_regressed_row(
        self, floor: ProtectedFloor, model: Any
    ) -> None:
        floor.regressions(candidate("breaks-all"))
        assert model.scored[3:] == ["e1"]

    def test_never_scores_rows_the_reference_failed(
        self, model: Any, rows: list[dict[str, str]]
    ) -> None:
        floor = ProtectedFloor(model, rows, ("emergency",))
        floor.measure(candidate("fails-e2"))
        floor.regressions(candidate("holds"))
        assert model.scored[3:] == ["e1"]

    def test_checks_previous_failures_first(
        self, floor: ProtectedFloor, model: Any
    ) -> None:
        floor.regressions(candidate("breaks-e2"))
        floor.regressions(candidate("breaks-all"))
        assert model.scored[5:] == ["e2"]

    @pytest.mark.parametrize("size", [1, 2])
    def test_rejects_wrong_result_count(
        self, rows: list[dict[str, str]], protected: tuple[str, ...], size: int
    ) -> None:
        config = FloorConfig(batch_size=size)
        floor = ProtectedFloor(lambda batch, _: [], rows, protected, config)
        with pytest.raises(ValueError, match=f"0 results, expected {size}"):
            floor.measure(candidate("seed"))

    def test_reports_every_check_of_the_row(
        self, two_check_model: Any, protected: tuple[str, ...]
    ) -> None:
        floor = ProtectedFloor(two_check_model, [{"id": "m1"}], protected)
        floor.measure(candidate("seed"))
        assert floor.regressions(candidate("breaks-both")) == (
            FloorRegression(0, "emergency"),
            FloorRegression(0, "encoding"),
        )


class TestCache:
    def test_repeat_candidate_is_free(self, floor: ProtectedFloor) -> None:
        floor.regressions(candidate("holds"))
        before = floor.evaluations
        floor.regressions(candidate("holds"))
        assert floor.evaluations == before

    def test_cache_off_scores_again(
        self, model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        floor = ProtectedFloor(model, rows, protected, FloorConfig(cache_size=0))
        floor.measure(candidate("seed"))
        floor.regressions(candidate("holds"))
        floor.regressions(candidate("holds"))
        assert floor.evaluations == 9

    def test_evicts_oldest(
        self, model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        floor = ProtectedFloor(model, rows, protected, FloorConfig(cache_size=1))
        floor.measure(candidate("seed"))
        for text in ("holds", "breaks-e2", "holds"):
            floor.regressions(candidate(text))
        assert floor.evaluations == 3 + 3 + 2 + 3


class TestThreads:
    def test_checks_do_not_overlap(
        self, meeting_model: Any, rows: list[dict[str, str]], side_by_side: Any
    ) -> None:
        floor = ProtectedFloor(meeting_model, rows, ("emergency",))
        floor.measure(candidate("seed"))
        meeting_model.start_meeting()
        side_by_side(floor.regressions, candidate("holds"), candidate("breaks-e2"))
        assert not meeting_model.meeting.overlapped


class TestBatches:
    def test_measure_scores_rows_in_batches(
        self, batched: ProtectedFloor, model: Any
    ) -> None:
        assert model.calls == [2, 1]

    def test_counts_rows_not_calls(self, batched: ProtectedFloor) -> None:
        assert batched.evaluations == 3

    def test_stops_after_the_first_regressed_batch(
        self, batched: ProtectedFloor, model: Any
    ) -> None:
        batched.regressions(candidate("breaks-all"))
        assert model.calls[2:] == [2]

    def test_reports_the_first_regressed_row(self, batched: ProtectedFloor) -> None:
        found = batched.regressions(candidate("breaks-all"))
        assert found == (FloorRegression(0, "emergency"),)

    def test_scores_every_batch_of_a_holding_candidate(
        self, batched: ProtectedFloor, model: Any
    ) -> None:
        assert batched.regressions(candidate("holds")) == ()
        assert model.calls[2:] == [2, 1]

    def test_puts_every_regressed_row_of_a_batch_first(
        self, model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        floor = ProtectedFloor(model, rows, protected, FloorConfig(batch_size=3))
        floor.measure(candidate("seed"))
        floor.regressions(candidate("breaks-e2-x1"))
        found = floor.regressions(candidate("breaks-e1-x1"))
        assert found == (FloorRegression(2, "encoding"),)


class TestFromAdapter:
    def test_reads_objective_scores(
        self, table_adapter: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        floor = ProtectedFloor.from_adapter(table_adapter, rows, protected)
        assert floor.measure(candidate("seed")) == {"emergency": 2, "encoding": 1}

    def test_asks_for_no_traces(
        self, table_adapter: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        ProtectedFloor.from_adapter(table_adapter, rows, protected).measure(
            candidate("seed")
        )
        assert set(table_adapter.traces) == {False}

    def test_uses_the_config(
        self, table_adapter: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        config = FloorConfig(batch_size=3)
        ProtectedFloor.from_adapter(table_adapter, rows, protected, config).measure(
            candidate("seed")
        )
        assert table_adapter.model.calls == [3]

    def test_rejects_an_adapter_without_objective_scores(
        self,
        silent_adapter: Any,
        rows: list[dict[str, str]],
        protected: tuple[str, ...],
    ) -> None:
        floor = ProtectedFloor.from_adapter(silent_adapter, rows, protected)
        with pytest.raises(ValueError, match="objective_scores"):
            floor.measure(candidate("seed"))


class TestUnscored:
    def test_unscored_reference_case_is_not_protected(
        self, model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        floor = ProtectedFloor(model, rows, protected)
        assert floor.measure(candidate("unscored-e1")) == {
            "emergency": 1,
            "encoding": 1,
        }

    def test_unscored_case_raises(self, floor: ProtectedFloor) -> None:
        match = "no score for 'emergency' at floor row 0"
        with pytest.raises(UnscoredCheckError, match=match):
            floor.regressions(candidate("unscored-e1"))

    def test_unscored_result_is_not_cached(self, floor: ProtectedFloor) -> None:
        for _ in range(2):
            with pytest.raises(UnscoredCheckError):
                floor.regressions(candidate("unscored-e1"))
        assert floor.evaluations == 3 + 1 + 1

    def test_regression_in_the_batch_takes_precedence(
        self, model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        floor = ProtectedFloor(model, rows, protected, FloorConfig(batch_size=3))
        floor.measure(candidate("seed"))
        expected = (FloorRegression(2, "encoding"),)
        assert floor.regressions(candidate("unscored-e1-breaks-x1")) == expected
