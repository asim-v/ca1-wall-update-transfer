"""Resumable parallel byte-range downloader with mandatory MD5 validation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
from pathlib import Path
import shutil
import urllib.request


CHUNK_BYTES = 1024 * 1024


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _download_part(
    url: str,
    path: Path,
    start: int,
    end: int,
) -> tuple[int, int]:
    expected = end - start + 1
    existing = path.stat().st_size if path.exists() else 0
    if existing > expected:
        raise RuntimeError(f"oversized partial file: {path}")
    if existing == expected:
        return start, expected

    request_start = start + existing
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={request_start}-{end}",
            "User-Agent": "ca1-boundary-geometry/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError(
                f"server ignored byte range {request_start}-{end}: "
                f"HTTP {response.status}"
            )
        with path.open("ab") as destination:
            while chunk := response.read(CHUNK_BYTES):
                destination.write(chunk)
    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(
            f"partial file has {actual} bytes; expected {expected}: {path}"
        )
    return start, actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--md5", required=True)
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument(
        "--only-part",
        type=int,
        help="download one zero-based range; assemble once all ranges exist",
    )
    argument = parser.parse_args()
    if argument.bytes <= 0 or argument.parts <= 0:
        raise ValueError("bytes and parts must be positive")
    expected_md5 = argument.md5.lower()

    if argument.output.exists():
        existing_md5 = _md5(argument.output)
        if (
            argument.output.stat().st_size == argument.bytes
            and existing_md5 == expected_md5
        ):
            print(
                f"already complete: {argument.output} "
                f"({argument.bytes} bytes, md5 {existing_md5})"
            )
            return
        raise RuntimeError(
            "output exists but does not match; choose a new output path"
        )

    part_directory = argument.output.with_name(
        argument.output.name + ".parts"
    )
    part_directory.mkdir(parents=True, exist_ok=True)
    width = (argument.bytes + argument.parts - 1) // argument.parts
    jobs: list[tuple[Path, int, int]] = []
    for index in range(argument.parts):
        start = index * width
        if start >= argument.bytes:
            break
        end = min(argument.bytes - 1, (index + 1) * width - 1)
        jobs.append((part_directory / f"part-{index:03d}", start, end))

    download_jobs = jobs
    if argument.only_part is not None:
        if not 0 <= argument.only_part < len(jobs):
            raise ValueError("only-part index is out of range")
        download_jobs = [jobs[argument.only_part]]

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(download_jobs)
    ) as pool:
        futures = [
            pool.submit(_download_part, argument.url, path, start, end)
            for path, start, end in download_jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            start, size = future.result()
            print(f"completed range at {start}: {size} bytes", flush=True)

    complete = all(
        path.exists() and path.stat().st_size == end - start + 1
        for path, start, end in jobs
    )
    if not complete:
        print("selected range complete; other ranges remain")
        return

    assembling = argument.output.with_name(
        argument.output.name + ".assembling"
    )
    with assembling.open("wb") as destination:
        for path, _, _ in jobs:
            with path.open("rb") as source:
                shutil.copyfileobj(source, destination, CHUNK_BYTES)
    if assembling.stat().st_size != argument.bytes:
        raise RuntimeError("assembled file has the wrong byte count")
    actual_md5 = _md5(assembling)
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"MD5 mismatch: expected {expected_md5}, got {actual_md5}"
        )
    os.replace(assembling, argument.output)
    print(
        f"verified: {argument.output} "
        f"({argument.bytes} bytes, md5 {actual_md5})"
    )


if __name__ == "__main__":
    main()
