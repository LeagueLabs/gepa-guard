"""Errors raised by gepa-guard."""


class UnguardedMergeError(RuntimeError):
    """gepa accepted a merged candidate that did not go through the guard.

    Parameters
    ----------
    candidate : int
        Index of the merged candidate in the gepa state.

    Attributes
    ----------
    candidate : int
        Index of the merged candidate in the gepa state.
    """

    def __init__(self, candidate: int) -> None:
        super().__init__(
            f"gepa accepted merge candidate {candidate} without the guard. "
            "Run gepa with use_merge=False."
        )
        self.candidate = candidate

    def __reduce__(self) -> tuple[type["UnguardedMergeError"], tuple[int]]:
        """Rebuild the error from its candidate index when unpickled."""
        return type(self), (self.candidate,)
