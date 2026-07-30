"""Write a compact schema/session report for one released animal file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ca1_geometry.arena import introduced_boundaries
from ca1_geometry.io import Mat73Animal


def md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("animal_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    argument = parser.parse_args()

    with Mat73Animal(argument.animal_file) as animal:
        sessions = []
        for index in range(animal.n_sessions):
            info = animal.session_info(index)
            sessions.append(
                {
                    "session": index + 1,
                    "environment": info.environment,
                    "blocked": list(info.blocked),
                    "frames": info.frames,
                    "registered_cells": info.registered_cells,
                    "position_min": list(info.position_min),
                    "position_max": list(info.position_max),
                    "introduced_boundary_count": len(
                        introduced_boundaries(info.blocked)
                    ),
                }
            )

        report = {
            "file": argument.animal_file.name,
            "bytes": argument.animal_file.stat().st_size,
            "md5": md5(argument.animal_file),
            "matlab_storage": "7.3/HDF5",
            "n_sessions": animal.n_sessions,
            "longitudinal_cells": animal.n_cells,
            "stored_sampling_shape": list(
                animal.file["maps/sampling"].shape
            ),
            "stored_smoothed_shape": list(
                animal.file["maps/smoothed"].shape
            ),
            "stored_unsmoothed_shape": list(
                animal.file["maps/unsmoothed"].shape
            ),
            "sessions": sessions,
        }

    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
