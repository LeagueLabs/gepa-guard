"""Rules for protected objective names and scores."""

from collections.abc import Iterable
from math import isfinite
from typing import Any


def protected_names(protected: Iterable[str]) -> tuple[str, ...]:
    """Return the protected objective names as a tuple.

    Parameters
    ----------
    protected : iterable of str
        Objective names to protect.

    Returns
    -------
    tuple of str
        The names, in the given order.

    Raises
    ------
    ValueError
        If `protected` is a string, holds no names, or holds an empty or
        non-string name.
    """
    if isinstance(protected, str | bytes):
        raise ValueError("protected must be a collection of names, not one string")
    names = tuple(protected)
    if not names:
        raise ValueError("protected must name at least one objective")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("protected must hold only nonempty strings")
    return names


def is_pass(score: object) -> bool:
    """Return whether a protected score is a pass.

    Parameters
    ----------
    score : object
        The score, or None when the objective is absent.

    Returns
    -------
    bool
        True only for a score equal to 1.0.
    """
    return score == 1.0


def is_valid_score(score: Any) -> bool:
    """Return whether a score is a finite number.

    Parameters
    ----------
    score : Any
        The score to test.

    Returns
    -------
    bool
        False for NaN, infinity and values that are not numbers.
    """
    try:
        return isfinite(score)
    except (TypeError, ValueError, OverflowError):
        return False
