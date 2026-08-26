"""Local embedding provider using FastEmbed.

Provides :class:`LocalEmbeddingProvider`, a real embedding provider that
runs entirely on-device via `FastEmbed <https://github.com/qdrant/fastembed>`_
and ONNX Runtime. No API key, no paid API, and no network access after the
initial model download.

Default model
-------------
``BAAI/bge-small-en-v1.5`` (384 dimensions, ~130 MB download on first use).

Configuration
-------------
- ``REPOLENS_LOCAL_EMBEDDING_MODEL`` — override the default model name.

The first call to :meth:`LocalEmbeddingProvider.embed_text` (or
:meth:`embed_texts`) triggers a one-time model download from Hugging Face.
After download the model is cached locally and subsequent runs are fully
offline.

No module-level network calls happen at import time.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from repolens.embeddings import EmbeddingConfigError, EmbeddingProvider, Vector

_DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_LOCAL_DIMENSIONS = 384


class LocalEmbeddingProvider:
    """Embedding provider backed by a local FastEmbed ONNX model.

    Example::

        provider = LocalEmbeddingProvider()
        vector = provider.embed_text("database connection pool")
        vectors = provider.embed_texts(["auth", "payments"])

    Or via environment::

        provider = LocalEmbeddingProvider.from_env()
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_LOCAL_MODEL,
        dimensions: int | None = None,
    ) -> None:
        if not model:
            raise EmbeddingConfigError("model must not be empty")
        self._model = model
        self._dimensions = dimensions or _DEFAULT_LOCAL_DIMENSIONS
        self._embedder = None

    @classmethod
    def from_env(cls) -> LocalEmbeddingProvider:
        """Construct from ``REPOLENS_LOCAL_EMBEDDING_MODEL`` environment variable."""
        model = os.environ.get("REPOLENS_LOCAL_EMBEDDING_MODEL", _DEFAULT_LOCAL_MODEL).strip()
        if not model:
            model = _DEFAULT_LOCAL_MODEL
        return cls(model=model)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _ensure_model(self) -> None:
        """Lazily initialise the FastEmbed TextEmbedding model."""
        if self._embedder is not None:
            return
        from fastembed import TextEmbedding

        self._embedder = TextEmbedding(model_name=self._model)

    def embed_text(self, text: str) -> Vector:
        """Embed a single string using the local ONNX model."""
        self._ensure_model()
        try:
            embeddings = list(self._embedder.embed([text]))
        except Exception as exc:
            raise EmbeddingConfigError(
                f"FastEmbed embedding failed for input: {exc}"
            ) from exc
        vec = embeddings[0]
        return tuple(float(x) for x in vec)

    def embed_texts(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        """Embed many strings, preserving input order."""
        if not texts:
            return ()
        self._ensure_model()
        try:
            embeddings = list(self._embedder.embed(list(texts)))
        except Exception as exc:
            raise EmbeddingConfigError(
                f"FastEmbed embedding failed for batch input: {exc}"
            ) from exc
        return tuple(tuple(float(x) for x in vec) for vec in embeddings)
