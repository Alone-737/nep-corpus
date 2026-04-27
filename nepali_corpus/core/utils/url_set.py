from __future__ import annotations

from typing import Iterable

class UrlSet:
    """Python URL set."""

    def __init__(self) -> None:
        self._urls: set[str] = set()

    def add(self, url: str) -> None:
        if url:
            self._urls.add(url)

    def add_many(self, urls: Iterable[str]) -> int:
        before = len(self._urls)
        for url in urls:
            if url:
                self._urls.add(url)
        return len(self._urls) - before

    def contains(self, url: str) -> bool:
        return url in self._urls

    def __len__(self) -> int:
        return len(self._urls)
