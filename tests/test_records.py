from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from gepa_guard import FloorRegression, Regression, VetoStats


class TestRegressionRecords:
    @pytest.mark.parametrize(
        "record", [Regression(0, "emergency"), FloorRegression(3, "encoding")]
    )
    def test_is_frozen(self, record: Regression | FloorRegression) -> None:
        with pytest.raises(FrozenInstanceError):
            record.check = "other"  # ty: ignore[invalid-assignment]

    def test_equal_when_fields_match(self) -> None:
        assert Regression(1, "emergency") == Regression(1, "emergency")


class TestVetoStats:
    def test_starts_at_zero(self) -> None:
        assert VetoStats().as_dict()["proposals"] == 0

    def test_vetoed_by_guard_counts_both_scopes(self) -> None:
        stats = VetoStats(vetoed_subsample=2, vetoed_floor=3, vetoed_unverifiable=7)
        assert stats.vetoed_by_guard == 5

    def test_as_dict_sorts_checks(self) -> None:
        stats = VetoStats(checks=Counter({"encoding": 1, "emergency": 2}))
        assert list(stats.as_dict()["checks"]) == ["emergency", "encoding"]

    def test_as_dict_keys(self) -> None:
        assert set(VetoStats().as_dict()) == {
            "proposals",
            "accepted",
            "vetoed_by_guard",
            "vetoed_subsample",
            "vetoed_floor",
            "vetoed_unverifiable",
            "vetoed_by_base",
            "checks",
        }
