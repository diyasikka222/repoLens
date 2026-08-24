"""Tests for semantic retrieval with the offline embedding provider."""

from pathlib import Path

import pytest

from repolens.evaluation import EvaluationCase, EvaluationRunner, Searcher
from repolens.embeddings import FakeEmbeddingProvider
from repolens.search import CodeSearcher
from repolens.semantic_search import SemanticSearcher, cosine_similarity


def write_file(tmp_path: Path, relative: str, source: str) -> None:
    file_path = tmp_path / relative
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(source, encoding="utf-8")


def paths_of(results):
    return [result.file_path for result in results]


class RecordingProvider(FakeEmbeddingProvider):
    """Fake provider that records what it was asked to embed."""

    def __init__(self) -> None:
        super().__init__()
        self.batched_documents: list[list[str]] = []
        self.embedded_queries: list[str] = []

    def embed_texts(self, texts):
        self.batched_documents.append(list(texts))
        return super().embed_texts(texts)

    def embed_text(self, text):
        self.embedded_queries.append(text)
        return super().embed_text(text)


# --- 5. Cosine similarity ------------------------------------------------------------


def test_cosine_similarity_hand_computed_cases() -> None:
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((2.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)
    assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)
    assert cosine_similarity((1.0, 1.0), (1.0, 0.0)) == pytest.approx(2 ** -0.5)


def test_cosine_similarity_with_zero_vector_is_zero_not_error() -> None:
    assert cosine_similarity((), (1.0, 2.0)) == 0.0
    assert cosine_similarity((1.0, 2.0), ()) == 0.0


# --- 6./7. Repository items and queries are embedded -------------------------------


def test_repository_documents_are_embedded_at_construction(tmp_path: Path) -> None:
    write_file(tmp_path, "payments/refund.py", "def refund_transaction():\n    pass\n")
    write_file(tmp_path, "users/create.py", "def create_user():\n    pass\n")
    provider = RecordingProvider()

    searcher = SemanticSearcher(tmp_path, provider)

    assert len(provider.batched_documents) == 1
    documents = provider.batched_documents[0]
    assert len(documents) == 2
    joined = "\n".join(documents)
    assert "path: payments/refund.py" in joined
    assert "refund_transaction" in joined
    assert provider.embedded_queries == []


def test_query_is_embedded_on_each_search(tmp_path: Path) -> None:
    write_file(tmp_path, "app.py", "value = 1\n")
    provider = RecordingProvider()
    searcher = SemanticSearcher(tmp_path, provider)

    searcher.search("refund payment")
    searcher.search("refund payment")

    assert provider.embedded_queries == ["refund payment", "refund payment"]


# --- 8. Relevant result ranks above irrelevant ----------------------------------------


def test_matching_file_ranks_above_unrelated_file(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "payments/refund.py",
        "def refund_transaction(card_number, amount_cents):\n"
        "    return True  # refunds a card payment\n",
    )
    write_file(
        tmp_path,
        "users/profile.py",
        "class UserProfile:\n"
        "    def update_avatar(self, image):\n"
        "        pass\n",
    )
    searcher = SemanticSearcher(tmp_path, FakeEmbeddingProvider())

    results = searcher.search("refund a card payment")

    assert results[0].file_path == Path("payments") / "refund.py"
    similarities = [result.similarity for result in results]
    assert similarities == sorted(similarities, reverse=True)


def test_prose_in_docstring_matches_beyond_identifier_names(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "support/policy.py",
        '"""What to do when a customer asks for money back."""\nPOLICY_ID = 7\n',
    )
    write_file(
        tmp_path,
        "core/engine.py",
        "class Engine:\n    def start(self):\n        pass\n",
    )
    searcher = SemanticSearcher(tmp_path, FakeEmbeddingProvider())

    results = searcher.search("money back policy")

    assert results[0].file_path == Path("support") / "policy.py"


# --- 9. Sorted by similarity --------------------------------------------------------------


def test_results_are_sorted_by_descending_similarity(tmp_path: Path) -> None:
    write_file(tmp_path, "a_all.py", "# refund payment card\nx = 1\n")
    write_file(tmp_path, "b_partial.py", "# payment\ny = 2\n")
    write_file(tmp_path, "c_none.py", "class Unrelated:\n    pass\n")
    searcher = SemanticSearcher(tmp_path, FakeEmbeddingProvider())

    results = searcher.search("refund payment card")

    assert paths_of(results) == [Path("a_all.py"), Path("b_partial.py")]
    assert results[0].similarity > results[1].similarity > 0.0


# --- 10. Limit -------------------------------------------------------------------------------


def test_limit_is_respected(tmp_path: Path) -> None:
    for number in range(12):
        write_file(tmp_path, f"items/item_{number:02d}.py", f"# widget {number}\n")
    searcher = SemanticSearcher(tmp_path, FakeEmbeddingProvider())

    assert len(searcher.search("widget")) == 10
    assert len(searcher.search("widget", limit=3)) == 3
    assert len(searcher.search("widget", limit=100)) == 12
    assert searcher.search("widget", limit=0) == []


# --- 11. Empty repository ----------------------------------------------------------------------


def test_empty_repository_is_handled(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("docs only\n", encoding="utf-8")

    searcher = SemanticSearcher(tmp_path, FakeEmbeddingProvider())

    assert searcher.search("anything at all") == []


# --- 12. Empty / unknown query --------------------------------------------------------------------


def test_empty_and_unknown_queries_return_no_results(tmp_path: Path) -> None:
    write_file(tmp_path, "billing/invoice.py", "class InvoiceCalculator:\n    pass\n")
    searcher = SemanticSearcher(tmp_path, FakeEmbeddingProvider())

    assert searcher.search("") == []
    assert searcher.search("   ") == []
    assert searcher.search("zzzqqqxyzzy") == []


# --- 13. Repository-relative paths ---------------------------------------------------------------


def test_paths_are_repository_relative(tmp_path: Path) -> None:
    write_file(tmp_path, "deep/nested/mod.py", "# treasure here\nx = 1\n")
    searcher = SemanticSearcher(tmp_path, FakeEmbeddingProvider())

    results = searcher.search("treasure")

    assert len(results) == 1
    assert results[0].file_path == Path("deep") / "nested" / "mod.py"
    assert not results[0].file_path.is_absolute()


# --- 14. Determinism across repeated searches ---------------------------------------------------------


def test_repeated_searches_produce_identical_results(tmp_path: Path) -> None:
    write_file(tmp_path, "aa/tool.py", "# tool for cutting\nvalue = 1\n")
    write_file(tmp_path, "bb/tool.py", "# tool for cutting\nvalue = 1\n")
    searcher = SemanticSearcher(tmp_path, FakeEmbeddingProvider())

    first = searcher.search("cutting tool", limit=5)
    second = searcher.search("cutting tool", limit=5)

    assert first == second
    assert first[0].similarity == first[1].similarity
    assert paths_of(first) == [Path("aa") / "tool.py", Path("bb") / "tool.py"]


def test_each_file_appears_at_most_once(tmp_path: Path) -> None:
    write_file(tmp_path, "dup.py", "# refund refund refund payment\nx = 1\n")
    searcher = SemanticSearcher(tmp_path, FakeEmbeddingProvider())

    results = searcher.search("refund payment")

    assert paths_of(results).count(Path("dup.py")) == 1


# --- Evaluation framework integration ------------------------------------------------------------------


def test_both_searchers_satisfy_the_searcher_protocol(tmp_path: Path) -> None:
    write_file(tmp_path, "app.py", "value = 1\n")

    assert isinstance(CodeSearcher(tmp_path), Searcher)
    assert isinstance(SemanticSearcher(tmp_path, FakeEmbeddingProvider()), Searcher)


def test_evaluation_runner_drives_semantic_searcher(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "auth/login.py",
        "def authenticate():\n    pass  # verifies a user login session\n",
    )
    write_file(tmp_path, "billing/tax.py", "def apply_tax(amount):\n    pass\n")
    cases = [
        EvaluationCase("user login session", ["auth/login.py"]),
        EvaluationCase("tax amount", ["billing/tax.py"]),
    ]

    runner = EvaluationRunner(
        tmp_path, searcher=SemanticSearcher(tmp_path, FakeEmbeddingProvider())
    )

    report = runner.evaluate(cases, k=3)
    assert report.mean_reciprocal_rank == 1.0
    assert report.mean_precision_at_k == pytest.approx(1.0)
    assert report.mean_recall_at_k == pytest.approx(1.0)
