"""Experiment helper: print lexical vs semantic rankings for a query.

Used during corpus authoring to find queries that distinguish the two
retrieval strategies. Not part of the shipped benchmark.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from repolens.embeddings import FakeEmbeddingProvider
from repolens.search import CodeSearcher
from repolens.semantic_search import SemanticSearcher

CORPUS = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "comparison_corpus"


def compare(query: str, k: int = 8) -> None:
    root = CORPUS
    lex = CodeSearcher(root)
    sem = SemanticSearcher(root, FakeEmbeddingProvider())

    print(f"\n=== QUERY: {query!r} ===")
    lex_res = lex.search(query, limit=k)
    sem_res = sem.search(query, limit=k)
    lex_paths = [(r.file_path.as_posix(), r.score) for r in lex_res]
    sem_paths = [(r.file_path.as_posix(), round(r.similarity, 4)) for r in sem_res]
    print(f"  lexical  : {lex_paths}")
    print(f"  semantic : {sem_paths}")


if __name__ == "__main__":
    for q in sys.argv[1:]:
        compare(q)