import pickle

from gepa_guard import UnguardedMergeError


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
