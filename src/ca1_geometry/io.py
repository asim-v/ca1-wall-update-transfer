"""Lazy reader for the released MATLAB 7.3/HDF5 animal files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SessionInfo:
    """Compact metadata for one recording day."""

    index: int
    environment: str
    blocked: tuple[int, ...]
    frames: int
    registered_cells: int
    position_min: tuple[float, float]
    position_max: tuple[float, float]


class Mat73Animal:
    """Context-managed, session-lazy access to one released animal file.

    Arrays are exposed in analysis orientation even though MATLAB/HDF5 reverses
    several documented dimensions:

    - position: ``(frame, xy)``
    - trace: ``(frame, cell)``
    - maps: ``(y, x)`` or ``(cell, y, x)``
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: h5py.File | None = None

    def __enter__(self) -> "Mat73Animal":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
            required = {
                "SFPs",
                "blocked",
                "centroids",
                "envs",
                "maps",
                "position",
                "trace",
            }
            missing = required.difference(self._file.keys())
            if missing:
                self.close()
                raise ValueError(f"missing required fields: {sorted(missing)}")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError("animal file is not open")
        return self._file

    @property
    def n_sessions(self) -> int:
        return int(self.file["trace"].shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.file["SFPs"].shape[1])

    def _referenced_dataset(self, field: str, session: int) -> h5py.Dataset:
        if not 0 <= session < self.n_sessions:
            raise IndexError("session index out of range")
        references = self.file[field]
        reference = (
            references[session, 0]
            if references.shape[0] == self.n_sessions
            else references[0, session]
        )
        return self.file[reference]

    def environment(self, session: int) -> str:
        dataset = self._referenced_dataset("envs", session)
        codepoint = np.asarray(dataset, dtype=np.uint32).ravel()
        return "".join(chr(int(value)) for value in codepoint)

    def blocked(self, session: int) -> tuple[int, ...]:
        dataset = self._referenced_dataset("blocked", session)
        value = np.asarray(dataset, dtype=np.int64).ravel()
        if value.size == 1 and value[0] == -1:
            return ()
        return tuple(int(item) for item in value)

    def position(self, session: int) -> FloatArray:
        return np.asarray(
            self._referenced_dataset("position", session), dtype=np.float64
        )

    def trace(
        self, session: int, cells: NDArray[np.int64] | None = None
    ) -> FloatArray:
        dataset = self._referenced_dataset("trace", session)
        if cells is None:
            return np.asarray(dataset, dtype=np.float64)
        index = np.asarray(cells, dtype=np.int64)
        if index.ndim != 1:
            raise ValueError("cells must be a one-dimensional index array")
        # h5py requires monotonically increasing fancy indices.
        if index.size and np.any(np.diff(index) <= 0):
            raise ValueError("cells must be unique and strictly increasing")
        return np.asarray(dataset[:, index], dtype=np.float64)

    def registered_cells(
        self, session: int, sample_rows: int = 11
    ) -> BoolArray:
        """Identify cells with finite traces without loading the full session."""

        dataset = self._referenced_dataset("trace", session)
        rows = np.unique(
            np.linspace(0, dataset.shape[0] - 1, sample_rows, dtype=int)
        )
        sample = np.asarray(dataset[rows, :], dtype=np.float64)
        return np.any(np.isfinite(sample), axis=0)

    def common_registered_cells(self, *sessions: int) -> NDArray[np.int64]:
        if not sessions:
            raise ValueError("at least one session is required")
        common = np.logical_and.reduce(
            [self.registered_cells(session) for session in sessions]
        )
        return np.flatnonzero(common)

    def sampling_map(self, session: int) -> FloatArray:
        return np.asarray(
            self.file["maps/sampling"][session], dtype=np.float64
        )

    def stored_rate_maps(
        self, session: int, *, smoothed: bool = False
    ) -> FloatArray:
        key = "smoothed" if smoothed else "unsmoothed"
        return np.asarray(self.file[f"maps/{key}"][session], dtype=np.float64)

    def session_info(self, session: int) -> SessionInfo:
        position = self.position(session)
        registered = self.registered_cells(session)
        return SessionInfo(
            index=session,
            environment=self.environment(session),
            blocked=self.blocked(session),
            frames=int(position.shape[0]),
            registered_cells=int(registered.sum()),
            position_min=tuple(
                float(value) for value in np.nanmin(position, axis=0)
            ),
            position_max=tuple(
                float(value) for value in np.nanmax(position, axis=0)
            ),
        )
