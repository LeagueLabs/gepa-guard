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


class UnscoredCheckError(RuntimeError):
    """A floor row has no score for a protected check that the reference passed.

    Parameters
    ----------
    row : int
        Index of the row in the floor.
    check : str
        Name of the protected check.

    Attributes
    ----------
    row : int
        Index of the row in the floor.
    check : str
        Name of the protected check.
    """

    def __init__(self, row: int, check: str) -> None:
        super().__init__(f"no score for {check!r} at floor row {row}")
        self.row = row
        self.check = check

    def __reduce__(self) -> tuple[type["UnscoredCheckError"], tuple[int, str]]:
        """Rebuild the error from its row and check when unpickled."""
        return type(self), (self.row, self.check)
