import pickle

from gepa_guard import UnguardedMergeError, UnscoredCheckError


class TestUnguardedMergeError:
    def test_is_a_runtime_error(self) -> None:
        assert issubclass(UnguardedMergeError, RuntimeError)

    def test_keeps_the_candidate(self) -> None:
        assert UnguardedMergeError(3).candidate == 3

    def test_says_how_to_fix_the_run(self) -> None:
        assert "use_merge=False" in str(UnguardedMergeError(3))

    def test_survives_pickling(self) -> None:
        error = pickle.loads(pickle.dumps(UnguardedMergeError(3)))
        assert (error.candidate, str(error)) == (3, str(UnguardedMergeError(3)))


class TestUnscoredCheckError:
    def test_is_a_runtime_error(self) -> None:
        assert issubclass(UnscoredCheckError, RuntimeError)

    def test_keeps_the_row_and_check(self) -> None:
        error = UnscoredCheckError(3, "emergency")
        assert (error.row, error.check) == (3, "emergency")

    def test_names_the_check_and_row(self) -> None:
        expected = "no score for 'emergency' at floor row 3"
        assert str(UnscoredCheckError(3, "emergency")) == expected

    def test_survives_pickling(self) -> None:
        error = pickle.loads(pickle.dumps(UnscoredCheckError(3, "emergency")))
        assert (error.row, error.check) == (3, "emergency")
