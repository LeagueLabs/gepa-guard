import math
from typing import Any

import pytest

from gepa_guard._scores import is_pass, is_valid_score, protected_names


class TestProtectedNames:
    def test_returns_tuple(self) -> None:
        assert protected_names(["emergency", "encoding"]) == ("emergency", "encoding")

    @pytest.mark.parametrize(
        "names",
        [iter(["emergency", "encoding"]), (n for n in ["emergency", "encoding"])],
        ids=["iterator", "generator"],
    )
    def test_reads_an_iterator_once(self, names: Any) -> None:
        assert protected_names(names) == ("emergency", "encoding")

    @pytest.mark.parametrize(
        "value",
        ["emergency", b"emergency", (), [], ("",), ("ok", 1), iter([]), iter([""])],
    )
    def test_rejects_invalid(self, value: Any) -> None:
        with pytest.raises(ValueError, match="protected"):
            protected_names(value)


class TestIsPass:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (1.0, True),
            (1, True),
            (0.999, False),
            (0.0, False),
            (None, False),
            ("1.0", False),
        ],
    )
    def test_only_exactly_one_passes(self, score: object, expected: bool) -> None:
        assert is_pass(score) is expected


class TestIsValidScore:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (1.0, True),
            (0, True),
            (math.nan, False),
            (math.inf, False),
            ("1.0", False),
            (None, False),
        ],
    )
    def test_accepts_only_finite_numbers(self, score: object, expected: bool) -> None:
        assert is_valid_score(score) is expected
