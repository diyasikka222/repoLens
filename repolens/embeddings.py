"""Embedding providers for semantic retrieval.

An :class:`EmbeddingProvider` turns text into fixed-length numeric vectors
so that a searcher can rank repository content against a query by vector
similarity. The interface is a structural :class:`typing.Protocol`: any
object exposing ``embed_text`` and ``embed_texts`` qualifies, so RepoLens is
never hard-wired to a specific vendor or model.

The module ships only a deterministic offline implementation,
:class:`FakeEmbeddingProvider`, which maps words into a fixed number of
dimensions via a stable hash (a "hashing trick" bag of words) and
L2-normalizes the result. It requires no network access, no API key, and no
third-party dependencies, which keeps the unit-test suite fully offline and
reproducible. It is a *test double*, not a quality baseline: it captures
word-overlap semantics only.

Plugging in a real provider later
---------------------------------
A real provider (for example a hosted embedding API or a local model) needs
to do exactly three things:

1. Implement the two ``EmbeddingProvider`` methods; ``embed_texts`` should
   batch its inputs for efficiency.
2. Never perform network calls or credential lookups at import time — only
   inside the methods when the caller constructs and uses it.
3. Be injected explicitly at the call site::

       provider = MyRealEmbeddingProvider(api_key=...)
       searcher = SemanticSearcher(repo_root, provider)

No RepoLens module inspects provider internals, so provider-specific
configuration stays entirely outside the core system.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

Vector = tuple[float, ...]

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A minimal text-to-vector interface.

    Vectors from one provider must all have the same length and are expected
    to be comparable with cosine similarity. Implementations must be
    deterministic: identical input text yields identical vectors.
    """

    def embed_text(self, text: str) -> Vector:
        """Embed a single piece of text."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        """Embed many pieces of text, preserving input order."""
        ...


def l2_normalize(vector: Sequence[float]) -> Vector:
    """Return the unit-length version of ``vector``; zero vectors stay zero."""
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return tuple(float(component) for component in vector)
    return tuple(component / norm for component in vector)


class FakeEmbeddingProvider:
    """Deterministic local embeddings via hashed bag of words.

    Each lowercase word of the input selects one dimension using a stable
    BLAKE2b hash, term occurrences accumulate in that dimension, and the
    result is L2-normalized. Properties:

    - no randomness, no network, no credentials;
    - identical text always yields the identical vector, even across
      processes (``hashlib`` is stable, unlike the salted builtin ``hash``);
    - word order is ignored (bag of words);
    - different texts usually yield different vectors, though hash
      collisions between words are possible and harmless for testing.
    """

    def __init__(self, dimensions: int = 1024) -> None:
        if dimensions < 1:
            raise ValueError(f"dimensions must be at least 1, got {dimensions}")
        self.dimensions = dimensions

    def _embed(self, text: str) -> Vector:
        counts = [0.0] * self.dimensions
        for word in _WORD_PATTERN.findall(text.lower()):
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            counts[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        return l2_normalize(counts)

    def embed_text(self, text: str) -> Vector:
        return self._embed(text)

    def embed_texts(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        """Embed in bulk; deliberately does not dispatch through
        :meth:`embed_text` so subclasses may override the two independently."""
        return tuple(self._embed(text) for text in texts)
