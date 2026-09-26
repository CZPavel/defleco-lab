"""Bounded chronological frame history."""

from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class FrameHistory(Generic[T]):
    def __init__(self, capacity: int = 32):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._items: deque[T] = deque(maxlen=capacity)

    def append(self, item: T) -> None:
        self._items.append(item)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def latest(self, offset: int = 0) -> T:
        if offset < 0 or offset >= len(self._items):
            raise IndexError("history offset unavailable")
        return self._items[-1 - offset]

    def pair(self, stride: int = 1) -> tuple[T, T]:
        if stride < 1:
            raise ValueError("stride must be positive")
        return self.latest(stride), self.latest(0)

    def last(self, count: int, stride: int = 1) -> list[T]:
        if count < 1 or stride < 1:
            raise ValueError("count and stride must be positive")
        return [self.latest(i * stride) for i in reversed(range(count))]
