import pytest

from gepa_guard._fingerprint import candidate_fingerprint


class TestCandidateFingerprint:
    def test_is_sha256_hex(self) -> None:
        assert len(candidate_fingerprint({"prompt": "text"})) == 64

    def test_ignores_component_order(self) -> None:
        first = candidate_fingerprint({"a": "1", "b": "2"})
        assert first == candidate_fingerprint({"b": "2", "a": "1"})

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ({"prompt": "a"}, {"prompt": "b"}),
            ({"prompt": "a"}, {"system": "a"}),
            ({"ab": "c"}, {"a": "bc"}),
            ({"a": "b\0c"}, {"a": "b", "c": ""}),
            ({}, {"": ""}),
        ],
    )
    def test_differs_for_different_candidates(
        self, left: dict[str, str], right: dict[str, str]
    ) -> None:
        assert candidate_fingerprint(left) != candidate_fingerprint(right)
