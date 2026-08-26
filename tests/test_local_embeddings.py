"""Offline tests for LocalEmbeddingProvider.

All tests mock FastEmbed so that no model download or network access is
required during normal pytest execution.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from repolens.embeddings import EmbeddingConfigError, EmbeddingProvider, Vector
from repolens.local_embeddings import LocalEmbeddingProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_DIM = 384


def _fake_embedding(texts: list[str]) -> list[np.ndarray]:
    """Deterministic fake embeddings based on text length."""
    return [np.ones(_FAKE_DIM, dtype=np.float32) * (len(t) % 7 + 1) for t in texts]


def _mock_fastembed():
    """Patch fastembed.TextEmbedding to avoid model download."""
    mock_cls = MagicMock()
    instance = MagicMock()
    instance.embed.side_effect = _fake_embedding
    mock_cls.return_value = instance
    return patch("fastembed.TextEmbedding", mock_cls)


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_provider_satisfies_embedding_provider_protocol(self) -> None:
        with _mock_fastembed():
            provider = LocalEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_embed_text_returns_vector(self) -> None:
        with _mock_fastembed():
            provider = LocalEmbeddingProvider()
            result = provider.embed_text("hello world")
        assert isinstance(result, tuple)
        assert len(result) == _FAKE_DIM
        assert all(isinstance(c, float) for c in result)

    def test_embed_texts_returns_tuple_of_vectors(self) -> None:
        with _mock_fastembed():
            provider = LocalEmbeddingProvider()
            result = provider.embed_texts(["hello", "world"])
        assert isinstance(result, tuple)
        assert len(result) == 2
        for vec in result:
            assert isinstance(vec, tuple)
            assert len(vec) == _FAKE_DIM


# ---------------------------------------------------------------------------
# 2. Configuration handling
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_default_model(self) -> None:
        provider = LocalEmbeddingProvider()
        assert provider._model == "BAAI/bge-small-en-v1.5"

    def test_custom_model(self) -> None:
        provider = LocalEmbeddingProvider(model="custom/model")
        assert provider._model == "custom/model"

    def test_empty_model_raises_config_error(self) -> None:
        with pytest.raises(EmbeddingConfigError, match="model must not be empty"):
            LocalEmbeddingProvider(model="")

    def test_from_env_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REPOLENS_LOCAL_EMBEDDING_MODEL", raising=False)
        provider = LocalEmbeddingProvider.from_env()
        assert provider._model == "BAAI/bge-small-en-v1.5"

    def test_from_env_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REPOLENS_LOCAL_EMBEDDING_MODEL", "custom/model")
        provider = LocalEmbeddingProvider.from_env()
        assert provider._model == "custom/model"

    def test_from_env_empty_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REPOLENS_LOCAL_EMBEDDING_MODEL", "")
        provider = LocalEmbeddingProvider.from_env()
        assert provider._model == "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# 3. Dimension reporting
# ---------------------------------------------------------------------------


class TestDimensions:
    def test_default_dimensions(self) -> None:
        provider = LocalEmbeddingProvider()
        assert provider.dimensions == 384

    def test_custom_dimensions(self) -> None:
        provider = LocalEmbeddingProvider(dimensions=128)
        assert provider.dimensions == 128

    def test_embed_text_vector_matches_dimensions(self) -> None:
        with _mock_fastembed():
            provider = LocalEmbeddingProvider()
            result = provider.embed_text("test")
        assert len(result) == provider.dimensions


# ---------------------------------------------------------------------------
# 4. Embedding behaviour
# ---------------------------------------------------------------------------


class TestEmbedding:
    def test_embed_texts_preserves_order(self) -> None:
        texts = ["alpha", "beta", "gamma"]
        with _mock_fastembed():
            provider = LocalEmbeddingProvider()
            results = provider.embed_texts(texts)
        assert len(results) == 3

    def test_empty_texts_returns_empty_tuple(self) -> None:
        with _mock_fastembed():
            provider = LocalEmbeddingProvider()
            result = provider.embed_texts([])
        assert result == ()

    def test_single_text_in_batch(self) -> None:
        with _mock_fastembed():
            provider = LocalEmbeddingProvider()
            result = provider.embed_texts(["only one"])
        assert len(result) == 1
        assert isinstance(result[0], tuple)


# ---------------------------------------------------------------------------
# 5. Model initialisation isolation
# ---------------------------------------------------------------------------


class TestModelInitialisation:
    def test_construction_does_not_download_model(self) -> None:
        with patch("fastembed.TextEmbedding") as mock_cls:
            provider = LocalEmbeddingProvider()
            mock_cls.assert_not_called()
            assert provider._embedder is None

    def test_model_downloaded_lazily_on_embed(self) -> None:
        with patch("fastembed.TextEmbedding") as mock_cls:
            instance = MagicMock()
            instance.embed.return_value = [np.ones(_FAKE_DIM)]
            mock_cls.return_value = instance

            provider = LocalEmbeddingProvider()
            assert provider._embedder is None

            provider.embed_text("trigger download")
            mock_cls.assert_called_once()
            assert provider._embedder is instance

    def test_model_not_reinitialised_on_subsequent_calls(self) -> None:
        with patch("fastembed.TextEmbedding") as mock_cls:
            instance = MagicMock()
            instance.embed.return_value = [np.ones(_FAKE_DIM)]
            mock_cls.return_value = instance

            provider = LocalEmbeddingProvider()
            provider.embed_text("first")
            provider.embed_text("second")
            mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_fastembed_runtime_error_wrapped(self) -> None:
        with patch("fastembed.TextEmbedding") as mock_cls:
            instance = MagicMock()
            instance.embed.side_effect = RuntimeError("ONNX model load failed")
            mock_cls.return_value = instance

            provider = LocalEmbeddingProvider()
            with pytest.raises(EmbeddingConfigError, match="FastEmbed embedding failed"):
                provider.embed_text("test")

    def test_fastembed_batch_error_wrapped(self) -> None:
        with patch("fastembed.TextEmbedding") as mock_cls:
            instance = MagicMock()
            instance.embed.side_effect = RuntimeError("batch failure")
            mock_cls.return_value = instance

            provider = LocalEmbeddingProvider()
            with pytest.raises(EmbeddingConfigError, match="FastEmbed embedding failed"):
                provider.embed_texts(["a", "b"])


# ---------------------------------------------------------------------------
# 7. Unit test (no network) — fake embedding provider still works
# ---------------------------------------------------------------------------


class TestFakeProviderUnchanged:
    def test_fake_provider_still_works(self) -> None:
        from repolens.embeddings import FakeEmbeddingProvider

        provider = FakeEmbeddingProvider(dimensions=64)
        vec = provider.embed_text("test input")
        assert len(vec) == 64
        assert all(isinstance(c, float) for c in vec)


# ---------------------------------------------------------------------------
# 8. Integration test — opt-in only, requires model download
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    """Requires network access for model download on first run.

    Run with: pytest -m integration
    """

    def test_real_embed_text(self) -> None:
        provider = LocalEmbeddingProvider()
        vec = provider.embed_text("database connection pool")
        assert len(vec) == 384
        assert all(isinstance(c, float) for c in vec)
        assert any(c != 0.0 for c in vec)

    def test_real_embed_texts_order(self) -> None:
        provider = LocalEmbeddingProvider()
        texts = ["authentication", "payment processing", "logging"]
        results = provider.embed_texts(texts)
        assert len(results) == 3
        assert all(len(v) == 384 for v in results)

    def test_real_provider_satisfies_protocol(self) -> None:
        provider = LocalEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)
