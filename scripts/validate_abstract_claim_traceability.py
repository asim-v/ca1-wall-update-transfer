"""Validate every machine-readable quantitative claim in the abstract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "results" / "abstract_quantitative_claim_traceability_v1.json"


def _pointer(document: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError(f"not a JSON pointer: {pointer}")
    value = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _expected_source_value(claim: dict[str, Any], values: list[Any]) -> float:
    transformation = str(claim["transformation"])
    if transformation.startswith("add 0.5"):
        return float(values[0]) + 0.5
    if transformation.startswith("cross_location - shifted_minus_exact"):
        return float(values[0]) - float(values[1])
    return float(values[0])


def main() -> None:
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    assert trace["significance_statement"]["present"] is False
    assert trace["significance_statement"]["quantitative_claims"] == []

    reward = trace["source_repositories"]["reward"]
    for relative, expected in reward["source_file_sha256"].items():
        path = ROOT / reward["local_root"] / relative
        assert path.is_file(), path
        assert _sha256(path) == expected, path

    source_cache: dict[Path, Any] = {}
    quantitative_claims = 0
    for sentence in trace["abstract_sentences"]:
        assert sentence["sentence_id"].startswith("A")
        assert sentence["text"].strip()
        for claim in sentence["quantitative_claims"]:
            quantitative_claims += 1
            path = ROOT / claim["source_file"]
            assert path.is_file(), path
            if path not in source_cache:
                source_cache[path] = json.loads(path.read_text(encoding="utf-8"))
            pointers = claim.get("json_pointers", [claim.get("json_pointer")])
            assert pointers and all(pointers), claim["claim"]
            values = [_pointer(source_cache[path], pointer) for pointer in pointers]
            if "source_value" in claim:
                observed = _expected_source_value(claim, values)
                assert math.isclose(
                    observed,
                    float(claim["source_value"]),
                    rel_tol=0,
                    abs_tol=1e-14,
                ), claim["claim"]
            if "source_value_fraction" in claim:
                assert math.isclose(
                    float(values[0]),
                    float(claim["source_value_fraction"]),
                    rel_tol=0,
                    abs_tol=1e-14,
                ), claim["claim"]

    assert quantitative_claims > 0
    assert trace["all_abstract_quantitative_claims_machine_traceable"] is True
    print(
        "TRACEABILITY_OK",
        f"sentences={len(trace['abstract_sentences'])}",
        f"quantitative_claims={quantitative_claims}",
        f"source_files={len(source_cache)}",
    )


if __name__ == "__main__":
    main()
