"""Tests for the strategy benchmarking infrastructure.

These tests cover :func:`repolens.evaluation.benchmark_strategy`, which
benchmarks the lexical, semantic, candidate-semantic, and hybrid retrieval
strategies over the comparison corpus. The corpus is designed so that
semantic retrieval can recover target files that lexical retrieval ranks
below a "lexical magnet" (a module whose identifiers reuse the query
vocabulary but whose body is unrelated). Tests assert:

- the semantic/lexical quality distinction exists (requirement #6),
- candidate-semantic tracks candidate coverage of the ground truth
  (requirement #8/#9),
- embedding-cost behavior is bounded,
- all results are deterministic and offline.
"""

import json
import shutil
from pathlib import Path

import pytest

from repolens.embeddings import FakeEmbeddingProvider
from repolens.evaluation import (
    BenchmarkResult,
    CaseMetrics,
    EvaluationCase,
    benchmark_strategy,
)

COMPARISON_DIR = Path(__file__).resolve().parent / "fixtures" / "comparison_corpus"
CASES_JSON_PATH = COMPARISON_DIR / "cases.json"

DEFAULT_PROVIDER = FakeEmbeddingProvider

# Query strings expected to demand semantic recovery of auth/identity.py
# because core/misnamed.py names functions check_credentials/check_user.
SEMANTIC_CREDENTIAL_QUERIES = {
    "where do we validate user credentials",
    "check user login credentials",
}

# The competitor that lexical ranks above auth/identity.py for the credential
# queries. It only computes parity bits despite its misleading names.
LEXICAL_MAGNET = Path("core/misnamed.py")


def _all_py_files(root: Path) -> list[Path]:
    return sorted(p.relative_to(root) for p in root.rglob("*.py") if p.is_file())


class RecordingProvider(FakeEmbeddingProvider):
    """Fake provider that records what it was asked to embed."""

    def __init__(self) -> None:
        super().__init__()
        self.batched_documents: list[list[str]] = []
        self.embedded_queries: list[str] = []
        self.total_documents_embedded = 0

    def embed_texts(self, texts):
        self.batched_documents.append(list(texts))
        self.total_documents_embedded += len(texts)
        return super().embed_texts(texts)

    def embed_text(self, text):
        self.embedded_queries.append(text)
        return super().embed_text(text)


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    target = tmp_path / "comparison-corpus"
    shutil.copytree(COMPARISON_DIR, target)
    return target


@pytest.fixture()
def cases() -> list[EvaluationCase]:
    payload = json.loads(CASES_JSON_PATH.read_text(encoding="utf-8"))
    return [
        EvaluationCase(query=item["query"], relevant_files=item["relevant_files"])
        for item in payload["cases"]
    ]


def _by_query(result: BenchmarkResult) -> dict[str, CaseMetrics]:
    return {m.query: m for m in result.case_metrics}


def _benchmark(
    strategy: str,
    corpus: Path,
    cases: list[EvaluationCase],
    provider=None,
    **kwargs,
) -> BenchmarkResult:
    return benchmark_strategy(
        strategy,
        corpus,
        cases,
        k=5,
        provider=provider if provider is not None else DEFAULT_PROVIDER(),
        **kwargs,
    )


# --- CaseMetrics & BenchmarkResult shape -------------------------------------------


def test_benchmark_strategy_returns_expected_structure(corpus, cases) -> None:
    result = _benchmark("lexical", corpus, cases)

    assert isinstance(result, BenchmarkResult)
    assert result.strategy == "lexical"
    assert result.k == 5
    assert len(result.case_metrics) == len(cases)
    for metric in result.case_metrics:
        assert isinstance(metric, CaseMetrics)
        assert 0.0 <= metric.precision_at_k <= 1.0
        assert 0.0 <= metric.recall_at_k <= 1.0
        assert 0.0 <= metric.reciprocal_rank <= 1.0
        assert len(metric.retrieved_files) <= 5


def test_benchmark_strategy_empty_cases_raises(corpus) -> None:
    with pytest.raises(ValueError):
        benchmark_strategy("lexical", corpus, [])


def test_benchmark_strategy_unknown_strategy_raises(corpus, cases) -> None:
    with pytest.raises(ValueError):
        benchmark_strategy("bogus", corpus, cases, provider=DEFAULT_PROVIDER())


# --- Corpus structural sanity ------------------------------------------------------


def test_comparison_corpus_has_enough_files(corpus) -> None:
    # Requirement #5: enough files for meaningful ranking differences.
    assert len(_all_py_files(corpus)) >= 8


def test_cases_are_explicit_and_deterministic(corpus, cases) -> None:
    # Requirement #4: every case names a real, existing target file.
    all_files = set(_all_py_files(corpus))
    for case in cases:
        assert case.query.strip()
        assert case.relevant_files
        assert case.relevant_files <= all_files


# --- Lexical baseline ----------------------------------------------------------------


def test_lexical_ranks_magnet_above_credential_target(corpus, cases) -> None:
    # Requirement #6: lexical does NOT put auth/identity.py first; the
    # misleadingly-named core/misnamed.py outranks it.
    result = _benchmark("lexical", corpus, cases)
    by_query = _by_query(result)
    for query in SEMANTIC_CREDENTIAL_QUERIES:
        metric = by_query[query]
        assert metric.first_relevant_rank is not None
        assert metric.first_relevant_rank > 1  # not ranked first
        assert metric.retrieved_files[0] == LEXICAL_MAGNET


def test_lexical_never_embeds(corpus, cases) -> None:
    provider = RecordingProvider()
    result = _benchmark("lexical", corpus, cases, provider=provider)

    assert all(m.embedded_count is None for m in result.case_metrics)
    assert provider.total_documents_embedded == 0


def test_lexical_has_no_candidate_count(corpus, cases) -> None:
    result = _benchmark("lexical", corpus, cases)

    assert all(m.candidate_count is None for m in result.case_metrics)
    assert all(m.relevant_in_candidates is None for m in result.case_metrics)


# --- Semantic (full) distinct from lexical -------------------------------------------------


def test_semantic_recovers_credential_target_where_lexical_fails(corpus, cases) -> None:
    # Requirement #6: semantic ranks auth/identity.py first even though a
    # misleadingly-named module outranks it lexically.
    result = _benchmark("semantic", corpus, cases)
    by_query = _by_query(result)
    for query in SEMANTIC_CREDENTIAL_QUERIES:
        metric = by_query[query]
        assert metric.first_relevant_rank == 1
        assert metric.reciprocal_rank == 1.0


def test_semantic_improves_mrr_over_lexical(corpus, cases) -> None:
    lex = _benchmark("lexical", corpus, cases)
    sem = _benchmark("semantic", corpus, cases)

    # The two credential queries are ranked 2nd (MRR 0.5 each) by lexical and
    # 1st (MRR 1.0 each) by semantic; the three balanced queries are already
    # 1.0 in both. So semantic mean MRR strictly exceeds lexical.
    assert sem.mean_reciprocal_rank > lex.mean_reciprocal_rank


def test_semantic_embeds_every_file_once(corpus, cases) -> None:
    provider = RecordingProvider()
    _benchmark("semantic", corpus, cases, provider=provider)

    total_files = len(_all_py_files(corpus))
    assert provider.total_documents_embedded == total_files


def test_semantic_records_embedded_count(corpus, cases) -> None:
    result = _benchmark("semantic", corpus, cases)

    counts = [m.embedded_count for m in result.case_metrics]
    assert all(c is not None for c in counts)
    assert all(c == counts[0] for c in counts)


# --- Candidate semantic --------------------------------------------------------------


def test_candidate_semantic_matches_full_semantic_at_ample_limit(corpus, cases) -> None:
    # Requirement #8: with limit 40 the candidate-semantic strategy covers every
    # file, so it must match full semantic on recall and MRR.
    full = _benchmark("semantic", corpus, cases)
    cand = _benchmark("candidate-semantic", corpus, cases, candidate_limit=40)

    assert cand.mean_recall_at_k == pytest.approx(full.mean_recall_at_k)
    assert cand.mean_reciprocal_rank == pytest.approx(full.mean_reciprocal_rank)
    assert cand.total_embedded <= full.total_embedded


def test_candidate_semantic_embeds_no_more_than_candidate_limit(corpus, cases) -> None:
    provider = RecordingProvider()
    result = _benchmark(
        "candidate-semantic", corpus, cases, provider=provider, candidate_limit=3
    )

    # Each individual search embeds at most candidate_limit new documents
    # (a single embed_texts batch); the cumulative cache may grow across
    # different queries.
    assert all(len(batch) <= 3 for batch in provider.batched_documents)
    assert all(m.candidate_count == 3 for m in result.case_metrics)


def test_candidate_semantic_embeds_fewer_than_full_semantic(corpus, cases) -> None:
    full_provider = RecordingProvider()
    candidate_provider = RecordingProvider()

    full = _benchmark("semantic", corpus, cases, provider=full_provider)
    candidate = _benchmark(
        "candidate-semantic",
        corpus,
        cases,
        provider=candidate_provider,
        candidate_limit=2,
    )

    assert candidate_provider.total_documents_embedded < full_provider.total_documents_embedded
    assert candidate.total_embedded < full.total_embedded


def test_candidate_semantic_reuses_embeddings_across_queries(corpus, cases) -> None:
    provider = RecordingProvider()

    _benchmark(
        "candidate-semantic", corpus, cases, provider=provider, candidate_limit=3
    )

    assert provider.total_documents_embedded == len({
        doc for batch in provider.batched_documents for doc in batch
    })


def test_tight_candidate_limit_drops_credential_target(corpus, cases) -> None:
    # Requirement #8/#9: a candidate limit smaller than the target's lexical
    # rank drops the ground-truth file and must be reported, not hidden.
    result = _benchmark(
        "candidate-semantic", corpus, cases, candidate_limit=1
    )
    by_query = _by_query(result)
    for query in SEMANTIC_CREDENTIAL_QUERIES:
        metric = by_query[query]
        # auth/identity.py is ranked 2nd lexically, so it is not a candidate
        # at limit 1 → the case cannot recover it and recall drops to 0.
        assert metric.relevant_in_candidates is False
        assert metric.recall_at_k == 0.0
        assert metric.reciprocal_rank == 0.0


def test_ample_candidate_limit_reports_coverage(corpus, cases) -> None:
    result = _benchmark("candidate-semantic", corpus, cases, candidate_limit=40)
    for metric in result.case_metrics:
        assert metric.relevant_in_candidates is True


# --- Hybrid --------------------------------------------------------------------------


def test_hybrid_uses_candidate_semantic(corpus, cases) -> None:
    provider = RecordingProvider()
    result = _benchmark("hybrid", corpus, cases, provider=provider)

    total_files = len(_all_py_files(corpus))
    assert result.total_embedded <= total_files
    assert result.total_embedded > 0
    assert all(m.embedded_count is not None for m in result.case_metrics)


def test_hybrid_recovers_every_case_at_ample_limit(corpus, cases) -> None:
    result = _benchmark("hybrid", corpus, cases, candidate_limit=40)
    # Hybrid retrieves every ground-truth target at an ample candidate limit
    # (its lexical leg surfaces the target even when semantic fusion does not
    # reorder it to the top).
    assert result.mean_recall_at_k == pytest.approx(1.0)
    for metric in result.case_metrics:
        assert metric.recall_at_k == pytest.approx(1.0)


# --- Aggregate metrics are consistent -----------------------------------------------


def test_aggregate_metrics_match_per_case(corpus, cases) -> None:
    result = _benchmark("candidate-semantic", corpus, cases)

    n = len(result.case_metrics)
    assert result.mean_precision_at_k == pytest.approx(
        sum(m.precision_at_k for m in result.case_metrics) / n
    )
    assert result.mean_recall_at_k == pytest.approx(
        sum(m.recall_at_k for m in result.case_metrics) / n
    )
    assert result.mean_reciprocal_rank == pytest.approx(
        sum(m.reciprocal_rank for m in result.case_metrics) / n
    )


def test_benchmark_is_deterministic(corpus, cases) -> None:
    first = _benchmark("candidate-semantic", corpus, cases)
    second = _benchmark("candidate-semantic", corpus, cases)

    for a, b in zip(first.case_metrics, second.case_metrics):
        assert a.query == b.query
        assert a.retrieved_files == b.retrieved_files
        assert a.relevant_files == b.relevant_files
        assert a.relevant_in_candidates == b.relevant_in_candidates
        assert a.precision_at_k == pytest.approx(b.precision_at_k)
        assert a.recall_at_k == pytest.approx(b.recall_at_k)
        assert a.reciprocal_rank == pytest.approx(b.reciprocal_rank)
    assert first.mean_precision_at_k == pytest.approx(second.mean_precision_at_k)
    assert first.mean_recall_at_k == pytest.approx(second.mean_recall_at_k)
    assert first.mean_reciprocal_rank == pytest.approx(second.mean_reciprocal_rank)