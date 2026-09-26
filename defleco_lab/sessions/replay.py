"""Pure helpers for reconstructing chronological replay history."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def replay_history(items: Sequence[T], current_index: int, requirement: int) -> list[T]:
    """Return the exact chronological source span ending at ``current_index``.

    An empty list represents a valid seek position with insufficient preceding
    history, rather than allowing a stale result from another position.
    """
    if requirement < 1:
        raise ValueError("history requirement must be positive")
    if not 0 <= current_index < len(items):
        raise IndexError("replay index outside session")
    start = current_index - requirement + 1
    if start < 0:
        return []
    return list(items[start : current_index + 1])
