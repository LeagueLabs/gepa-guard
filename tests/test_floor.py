from typing import Any

import pytest

from gepa_guard import FloorConfig, FloorRegression, ProtectedFloor


@pytest.fixture
def floor(
    model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
) -> ProtectedFloor:
    floor = ProtectedFloor(model, rows, protected)
    floor.measure({"prompt": "seed"})
    return floor


def candidate(text: str) -> dict[str, str]:
    return {"prompt": text}


class TestFloorConfig:
    def test_default_cache_size(self) -> None:
        assert FloorConfig().cache_size == 128

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

    def test_scores_each_row_once(self, floor: ProtectedFloor, model: Any) -> None:
        assert model.scored == ["e1", "e2", "x1"]

    def test_clears_the_cache(self, floor: ProtectedFloor, model: Any) -> None:
        floor.regressions(candidate("holds"))
        floor.measure(candidate("seed"))
        floor.regressions(candidate("holds"))
        assert floor.evaluations == 3 + 3 + 3 + 3


class TestRegressions:
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

    def test_rejects_wrong_result_count(
        self, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> None:
        floor = ProtectedFloor(lambda batch, _: [], rows, protected)
        with pytest.raises(ValueError, match="1 row"):
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
