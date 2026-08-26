"""Embedding providers for semantic retrieval.

An :class:`EmbeddingProvider` turns text into fixed-length numeric vectors
so that a searcher can rank repository content against a query by vector
similarity. The interface is a structural :class:`typing.Protocol`: any
object exposing ``embed_text`` and ``embed_texts`` qualifies, so RepoLens is
never hard-wired to a specific vendor or model.

The module ships two implementations:

1. :class:`FakeEmbeddingProvider` — a deterministic offline test double that
   maps words into a fixed number of dimensions via a stable hash (a "hashing
   trick" bag of words) and L2-normalizes the result. It requires no network
   access, no API key, and no third-party dependencies, which keeps the
   unit-test suite fully offline and reproducible.

2. :class:`OpenAIEmbeddingProvider` — a real provider that calls any
   OpenAI-compatible embedding endpoint via the standard library. It uses
   environment variables for credentials and configuration, performs no
   network calls at import time, and batches requests for efficiency.

Configuration (real provider)
-----------------------------
Environment variables consumed by :class:`OpenAIEmbeddingProvider`:

- ``REPOLENS_EMBEDDING_API_KEY`` — API key for the embedding service
  (required).
- ``REPOLENS_EMBEDDING_MODEL`` — model identifier, e.g.
  ``"text-embedding-3-small"`` (required).
- ``REPOLENS_EMBEDDING_BASE_URL`` — base URL of the API, e.g.
  ``"https://api.openai.com"``. Defaults to ``"https://api.openai.com"``.
- ``REPOLENS_EMBEDDING_DIMENSIONS`` — optional output dimension count for
  models that support dimension reduction.

No API keys appear in source code or test output. Tests mock the HTTP layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
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


# ---------------------------------------------------------------------------
# Real embedding provider — OpenAI-compatible API
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "https://api.openai.com"
_EMBEDDINGS_PATH = "/v1/embeddings"
_DEFAULT_BATCH_SIZE = 2048


class EmbeddingProviderError(Exception):
    """Raised when the embedding provider encounters an unrecoverable error."""


class EmbeddingConfigError(EmbeddingProviderError):
    """Raised for missing or invalid configuration (credentials, model, etc.)."""


class EmbeddingAPIError(EmbeddingProviderError):
    """Raised when the remote API returns an error response."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class OpenAIEmbeddingProvider:
    """Embedding provider that calls an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Uses only the standard library (``urllib.request``) — no third-party
    HTTP clients required. Network calls happen exclusively inside
    :meth:`embed_text` / :meth:`embed_texts`; construction is cheap.

    Example::

        provider = OpenAIEmbeddingProvider.from_env()
        searcher = SemanticSearcher(repo_root, provider)

    Or with explicit arguments::

        provider = OpenAIEmbeddingProvider(
            api_key="sk-...",
            model="text-embedding-3-small",
            base_url="https://api.openai.com",
        )
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = _DEFAULT_BASE_URL,
        dimensions: int | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        if not api_key:
            raise EmbeddingConfigError("api_key must not be empty")
        if not model:
            raise EmbeddingConfigError("model must not be empty")
        if batch_size < 1:
            raise EmbeddingConfigError(f"batch_size must be at least 1, got {batch_size}")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._cached_dimensions: int | None = None

    @classmethod
    def from_env(cls) -> OpenAIEmbeddingProvider:
        """Construct from ``REPOLENS_EMBEDDING_*`` environment variables.

        Raises :class:`EmbeddingConfigError` if required variables are missing.
        """
        api_key = os.environ.get("REPOLENS_EMBEDDING_API_KEY", "")
        model = os.environ.get("REPOLENS_EMBEDDING_MODEL", "")
        base_url = os.environ.get("REPOLENS_EMBEDDING_BASE_URL", _DEFAULT_BASE_URL)
        dimensions_raw = os.environ.get("REPOLENS_EMBEDDING_DIMENSIONS", "")
        dimensions = int(dimensions_raw) if dimensions_raw else None

        if not api_key:
            raise EmbeddingConfigError(
                "REPOLENS_EMBEDDING_API_KEY environment variable is not set"
            )
        if not model:
            raise EmbeddingConfigError(
                "REPOLENS_EMBEDDING_MODEL environment variable is not set"
            )
        return cls(api_key=api_key, model=model, base_url=base_url, dimensions=dimensions)

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality.

        Determined from the first API call and cached; raises if the API
        has not been called yet.
        """
        if self._cached_dimensions is None:
            raise EmbeddingConfigError(
                "dimensions not yet known; call embed_text first or "
                "set dimensions explicitly in the constructor"
            )
        return self._cached_dimensions

    def embed_text(self, text: str) -> Vector:
        """Embed a single string by calling the API."""
        vectors = self._call_api([text])
        return vectors[0]

    def embed_texts(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        """Embed many strings, batching to respect API limits."""
        if not texts:
            return ()
        all_vectors: list[Vector] = []
        for offset in range(0, len(texts), self._batch_size):
            batch = texts[offset : offset + self._batch_size]
            all_vectors.extend(self._call_api(batch))
        return tuple(all_vectors)

    # -- internal -------------------------------------------------------------

    def _call_api(self, input_texts: list[str]) -> list[Vector]:
        """Send one ``/v1/embeddings`` request and extract vectors."""
        payload: dict[str, object] = {
            "model": self._model,
            "input": input_texts,
        }
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions

        url = f"{self._base_url}{_EMBEDDINGS_PATH}"
        body_bytes = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            message = f"HTTP {exc.code} from embedding API: {exc.reason}"
            if error_body:
                message += f": {error_body}"
            raise EmbeddingAPIError(
                message,
                status_code=exc.code,
                body=error_body,
            ) from exc
        except urllib.error.URLError as exc:
            raise EmbeddingAPIError(
                f"Failed to connect to embedding API: {exc.reason}"
            ) from exc

        return self._parse_response(response_body, len(input_texts))

    def _parse_response(self, response_body: str, expected_count: int) -> list[Vector]:
        """Parse the JSON response and extract embedding vectors."""
        try:
            data = json.loads(response_body)
        except (json.JSONDecodeError, ValueError) as exc:
            raise EmbeddingAPIError(
                f"Invalid JSON in API response: {exc}"
            ) from exc

        if "data" not in data or not isinstance(data["data"], list):
            raise EmbeddingAPIError("API response missing 'data' field or not a list")

        raw_embeddings = data["data"]
        if len(raw_embeddings) != expected_count:
            raise EmbeddingAPIError(
                f"Expected {expected_count} embeddings, got {len(raw_embeddings)}"
            )

        # Cache dimensions from the first response if not set
        if raw_embeddings and "embedding" in raw_embeddings[0]:
            first_vec = raw_embeddings[0]["embedding"]
            if self._cached_dimensions is None:
                self._cached_dimensions = len(first_vec)

        vectors: list[Vector] = []
        for item in raw_embeddings:
            if "embedding" not in item or not isinstance(item["embedding"], list):
                raise EmbeddingAPIError(
                    "Embedding entry missing 'embedding' field or not a list"
                )
            vectors.append(tuple(float(x) for x in item["embedding"]))
        return vectors
