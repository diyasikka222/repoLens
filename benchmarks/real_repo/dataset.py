"""Loading and validation of the manually curated benchmark queries.

This module is fully offline: it reads :data:`config.QUERIES_PATH` and turns
it into :class:`repolens.evaluation.EvaluationCase` objects. Ground truth is
stored by hand in ``queries.json``; nothing here runs the search algorithms
or inspects retrieval output.
"""

from __future__ import annotations

import json
from typing import Iterable

from repolens.evaluation import EvaluationCase

from benchmarks.real_repo.config import QUERIES_PATH


def load_cases() -> list[EvaluationCase]:
    """Load every evaluation case from ``queries.json``, preserving order."""
    payload = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    return [
        EvaluationCase(
            query=item["query"],
            relevant_files=item["relevant_files"],
        )
        for item in payload["cases"]
    ]


def validate_datasets_disjoint(cases: Iterable[EvaluationCase]) -> bool:
    """Return True if every ground-truth set is non-empty.

    (Small guard used by tests; keeps the dataset well-formed.)
    """
    return all(bool(case.relevant_files) for case in cases)
