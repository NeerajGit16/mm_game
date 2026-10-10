"""The storage layer: the only place the game touches files.

Replay files, coaching records, markdown summaries and the track record all
go through a ``Storage``. Names are relative, slash-separated paths --
``sessions/120m-7-0001/replay.json`` -- never filesystem paths, so the same
calls can later land in a browser download or local storage instead of a
directory. ``DirectoryStorage`` writes under one root; ``MemoryStorage``
keeps everything in a dict, for tests and for a front end that runs in the
page.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Protocol


class Storage(Protocol):
    def read_text(self, name: str) -> str: ...
    def write_text(self, name: str, text: str) -> None: ...
    def exists(self, name: str) -> bool: ...
    def names(self, prefix: str = "") -> list[str]: ...


def _check(name: str) -> PurePosixPath:
    """Refuse anything that could escape the store: absolute paths, parent
    references, backslashes, empty parts."""
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"not a valid storage name: {name!r}")
    return path


class MemoryStorage:
    def __init__(self) -> None:
        self._files: dict[str, str] = {}

    def read_text(self, name: str) -> str:
        _check(name)
        return self._files[name]

    def write_text(self, name: str, text: str) -> None:
        _check(name)
        self._files[name] = text

    def exists(self, name: str) -> bool:
        _check(name)
        return name in self._files

    def names(self, prefix: str = "") -> list[str]:
        return sorted(n for n in self._files if n.startswith(prefix))


class DirectoryStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, name: str) -> Path:
        return self.root.joinpath(*_check(name).parts)

    def read_text(self, name: str) -> str:
        return self._path(name).read_text(encoding="utf-8")

    def write_text(self, name: str, text: str) -> None:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    def exists(self, name: str) -> bool:
        return self._path(name).is_file()

    def names(self, prefix: str = "") -> list[str]:
        if not self.root.exists():
            return []
        found = (p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())
        return sorted(n for n in found if n.startswith(prefix))
