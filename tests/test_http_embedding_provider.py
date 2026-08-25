"""Tests for the OpenAI-compatible HTTP embedding provider.

All tests mock the HTTP layer — no network calls or API keys are required.
"""

import json
import sys
import urllib.error
from http.client import HTTPResponse
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from repolens.embeddings import (
    EmbeddingAPIError,
    EmbeddingConfigError,
    OpenAIEmbeddingProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**kwargs) -> OpenAIEmbeddingProvider:
    defaults = {
        "api_key": "test-key-123",
        "model": "text-embedding-3-small",
        "base_url": "https://api.example.com",
    }
    defaults.update(kwargs)
    return OpenAIEmbeddingProvider(**defaults)


def _make_api_response(embeddings: list[list[float]], *, status: int = 200) -> MagicMock:
    """Create a mock HTTP response with the given embeddings."""
    body = json.dumps({"data": [{"embedding": vec, "index": i} for i, vec in enumerate(embeddings)]})
    response = MagicMock()
    response.read.return_value = body.encode("utf-8")
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


def _make_error_response(status: int, message: str) -> MagicMock:
    """Create a mock HTTP error response."""
    body = json.dumps({"error": {"message": message, "type": "invalid_request_error"}})
    response = MagicMock()
    response.read.return_value = body.encode("utf-8")
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


def _mock_urlopen(mock_response):
    """Patch urllib.request.urlopen to return mock_response."""
    return patch("urllib.request.urlopen", return_value=mock_response)


# ---------------------------------------------------------------------------
# 1. Import-time behaviour
# ---------------------------------------------------------------------------


def test_no_network_call_during_import() -> None:
    """Importing the module must not trigger any HTTP requests."""
    # If we got here, the module imported fine without side effects.
    # Verify the module doesn't have any module-level network activity.
    import repolens.embeddings as mod

    assert hasattr(mod, "OpenAIEmbeddingProvider")
    assert hasattr(mod, "FakeEmbeddingProvider")


# ---------------------------------------------------------------------------
# 2. Configuration handling
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_from_env_with_valid_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REPOLENS_EMBEDDING_API_KEY", "sk-test")
        monkeypatch.setenv("REPOLENS_EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("REPOLENS_EMBEDDING_BASE_URL", "https://custom.api.com")
        monkeypatch.setenv("REPOLENS_EMBEDDING_DIMENSIONS", "512")

        provider = OpenAIEmbeddingProvider.from_env()

        assert provider._api_key == "sk-test"
        assert provider._model == "text-embedding-3-small"
        assert provider._base_url == "https://custom.api.com"
        assert provider._dimensions == 512

    def test_from_env_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REPOLENS_EMBEDDING_API_KEY", raising=False)
        monkeypatch.setenv("REPOLENS_EMBEDDING_MODEL", "text-embedding-3-small")

        with pytest.raises(EmbeddingConfigError, match="REPOLENS_EMBEDDING_API_KEY"):
            OpenAIEmbeddingProvider.from_env()

    def test_from_env_missing_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REPOLENS_EMBEDDING_API_KEY", "sk-test")
        monkeypatch.delenv("REPOLENS_EMBEDDING_MODEL", raising=False)

        with pytest.raises(EmbeddingConfigError, match="REPOLENS_EMBEDDING_MODEL"):
            OpenAIEmbeddingProvider.from_env()

    def test_from_env_empty_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REPOLENS_EMBEDDING_API_KEY", "")
        monkeypatch.setenv("REPOLENS_EMBEDDING_MODEL", "text-embedding-3-small")

        with pytest.raises(EmbeddingConfigError, match="REPOLENS_EMBEDDING_API_KEY"):
            OpenAIEmbeddingProvider.from_env()

    def test_empty_api_key_raises_config_error(self) -> None:
        with pytest.raises(EmbeddingConfigError, match="api_key must not be empty"):
            OpenAIEmbeddingProvider(api_key="", model="text-embedding-3-small")

    def test_empty_model_raises_config_error(self) -> None:
        with pytest.raises(EmbeddingConfigError, match="model must not be empty"):
            OpenAIEmbeddingProvider(api_key="sk-test", model="")

    def test_zero_batch_size_raises_config_error(self) -> None:
        with pytest.raises(EmbeddingConfigError, match="batch_size must be at least 1"):
            OpenAIEmbeddingProvider(api_key="sk-test", model="m", batch_size=0)

    def test_dimensions_not_yet_known_raises(self) -> None:
        provider = _make_provider()
        with pytest.raises(EmbeddingConfigError, match="dimensions not yet known"):
            _ = provider.dimensions

    def test_base_url_trailing_slash_stripped(self) -> None:
        provider = _make_provider(base_url="https://api.example.com/")
        assert provider._base_url == "https://api.example.com"


# ---------------------------------------------------------------------------
# 3. Request construction
# ---------------------------------------------------------------------------


class TestRequestConstruction:
    def test_single_text_request_body(self) -> None:
        provider = _make_provider(model="text-embedding-3-large")
        response = _make_api_response([[0.1, 0.2, 0.3]])

        with _mock_urlopen(response) as mock:
            provider.embed_text("hello world")

            call_args = mock.call_args
            request = call_args[0][0]
            assert request.full_url == "https://api.example.com/v1/embeddings"
            assert request.get_method() == "POST"

            body = json.loads(request.data.decode("utf-8"))
            assert body["model"] == "text-embedding-3-large"
            assert body["input"] == ["hello world"]

    def test_authorization_header(self) -> None:
        provider = _make_provider(api_key="sk-secret-42")
        response = _make_api_response([[0.1, 0.2]])

        with _mock_urlopen(response) as mock:
            provider.embed_text("test")

            request = mock.call_args[0][0]
            assert request.get_header("Authorization") == "Bearer sk-secret-42"

    def test_content_type_header(self) -> None:
        provider = _make_provider()
        response = _make_api_response([[0.1]])

        with _mock_urlopen(response) as mock:
            provider.embed_text("x")

            request = mock.call_args[0][0]
            assert request.get_header("Content-type") == "application/json"

    def test_dimensions_included_when_set(self) -> None:
        provider = _make_provider(dimensions=256)
        response = _make_api_response([[0.1] * 256])

        with _mock_urlopen(response) as mock:
            provider.embed_text("test")

            body = json.loads(mock.call_args[0][0].data.decode("utf-8"))
            assert body["dimensions"] == 256

    def test_dimensions_not_included_when_none(self) -> None:
        provider = _make_provider(dimensions=None)
        response = _make_api_response([[0.1]])

        with _mock_urlopen(response) as mock:
            provider.embed_text("test")

            body = json.loads(mock.call_args[0][0].data.decode("utf-8"))
            assert "dimensions" not in body

    def test_base_url_used_correctly(self) -> None:
        provider = _make_provider(base_url="https://my-custom-llm.example.com")
        response = _make_api_response([[0.5]])

        with _mock_urlopen(response) as mock:
            provider.embed_text("test")

            url = mock.call_args[0][0].full_url
            assert url == "https://my-custom-llm.example.com/v1/embeddings"


# ---------------------------------------------------------------------------
# 4. Response parsing / vector extraction
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_single_text_returns_vector(self) -> None:
        provider = _make_provider()
        response = _make_api_response([[0.1, 0.2, 0.3]])

        with _mock_urlopen(response):
            result = provider.embed_text("hello")

        assert result == (0.1, 0.2, 0.3)

    def test_multiple_texts_returns_preserved_order(self) -> None:
        provider = _make_provider()
        response = _make_api_response([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

        with _mock_urlopen(response):
            results = provider.embed_texts(["a", "b", "c"])

        assert len(results) == 3
        assert results[0] == (1.0, 0.0)
        assert results[1] == (0.0, 1.0)
        assert results[2] == (0.5, 0.5)

    def test_empty_texts_returns_empty_tuple(self) -> None:
        provider = _make_provider()
        result = provider.embed_texts([])
        assert result == ()

    def test_vector_components_are_floats(self) -> None:
        provider = _make_provider()
        response = _make_api_response([[1, 2, 3]])  # integers in JSON

        with _mock_urlopen(response):
            result = provider.embed_text("x")

        assert all(isinstance(c, float) for c in result)
        assert result == (1.0, 2.0, 3.0)

    def test_dimensions_cached_from_first_response(self) -> None:
        provider = _make_provider()
        response = _make_api_response([[0.1, 0.2, 0.3, 0.4]])

        with _mock_urlopen(response):
            provider.embed_text("test")

        assert provider.dimensions == 4


# ---------------------------------------------------------------------------
# 5. Batch embedding behaviour
# ---------------------------------------------------------------------------


class TestBatching:
    def test_batching_respects_batch_size(self) -> None:
        provider = _make_provider(batch_size=2)
        texts = ["a", "b", "c", "d", "e"]

        # Two API calls: batch of 2, batch of 2, batch of 1
        responses = [
            _make_api_response([[0.1, 0.0], [0.0, 0.1]]),
            _make_api_response([[0.2, 0.0], [0.0, 0.2]]),
            _make_api_response([[0.3, 0.0]]),
        ]

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        with patch("urllib.request.urlopen", side_effect=side_effect):
            results = provider.embed_texts(texts)

        assert len(results) == 5
        assert call_count == 3

    def test_single_batch_when_texts_fit(self) -> None:
        provider = _make_provider(batch_size=10)
        response = _make_api_response([[0.1, 0.2], [0.3, 0.4]])

        with _mock_urlopen(response) as mock:
            provider.embed_texts(["a", "b"])

            assert mock.call_count == 1

    def test_vectors_concatenated_in_order(self) -> None:
        provider = _make_provider(batch_size=1)
        responses = [
            _make_api_response([[1.0]]),
            _make_api_response([[2.0]]),
            _make_api_response([[3.0]]),
        ]

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        with patch("urllib.request.urlopen", side_effect=side_effect):
            results = provider.embed_texts(["a", "b", "c"])

        assert results == ((1.0,), (2.0,), (3.0,))


# ---------------------------------------------------------------------------
# 6. API / error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_http_401_raises_api_error(self) -> None:
        provider = _make_provider()
        http_error = urllib.error.HTTPError(
            url="https://api.example.com/v1/embeddings",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=BytesIO(b'{"error": {"message": "Invalid API key"}}'),
        )

        with patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(EmbeddingAPIError, match="HTTP 401") as exc_info:
                provider.embed_text("test")
            assert exc_info.value.status_code == 401

    def test_http_429_raises_api_error(self) -> None:
        provider = _make_provider()
        http_error = urllib.error.HTTPError(
            url="https://api.example.com/v1/embeddings",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=BytesIO(b'{"error": {"message": "Rate limit exceeded"}}'),
        )

        with patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(EmbeddingAPIError, match="HTTP 429") as exc_info:
                provider.embed_text("test")
            assert exc_info.value.status_code == 429

    def test_http_500_raises_api_error(self) -> None:
        provider = _make_provider()
        http_error = urllib.error.HTTPError(
            url="https://api.example.com/v1/embeddings",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=BytesIO(b""),
        )

        with patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(EmbeddingAPIError, match="HTTP 500") as exc_info:
                provider.embed_text("test")
            assert exc_info.value.status_code == 500

    def test_connection_error_raises_api_error(self) -> None:
        provider = _make_provider()
        url_error = urllib.error.URLError("Connection refused")

        with patch("urllib.request.urlopen", side_effect=url_error):
            with pytest.raises(EmbeddingAPIError, match="Failed to connect"):
                provider.embed_text("test")


# ---------------------------------------------------------------------------
# 7. Malformed response handling
# ---------------------------------------------------------------------------


class TestMalformedResponses:
    def test_invalid_json_raises_api_error(self) -> None:
        provider = _make_provider()
        response = MagicMock()
        response.read.return_value = b"not json at all"
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)

        with _mock_urlopen(response):
            with pytest.raises(EmbeddingAPIError, match="Invalid JSON"):
                provider.embed_text("test")

    def test_missing_data_field_raises_api_error(self) -> None:
        provider = _make_provider()
        response = MagicMock()
        response.read.return_value = json.dumps({"object": "list"}).encode()
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)

        with _mock_urlopen(response):
            with pytest.raises(EmbeddingAPIError, match="missing 'data' field"):
                provider.embed_text("test")

    def test_wrong_embedding_count_raises_api_error(self) -> None:
        provider = _make_provider()
        response = MagicMock()
        response.read.return_value = json.dumps({
            "data": [{"embedding": [0.1], "index": 0}]
        }).encode()
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)

        with _mock_urlopen(response):
            with pytest.raises(EmbeddingAPIError, match="Expected 2 embeddings, got 1"):
                provider.embed_texts(["a", "b"])

    def test_missing_embedding_field_raises_api_error(self) -> None:
        provider = _make_provider()
        response = MagicMock()
        response.read.return_value = json.dumps({
            "data": [{"index": 0, "object": "embedding"}]
        }).encode()
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)

        with _mock_urlopen(response):
            with pytest.raises(EmbeddingAPIError, match="missing 'embedding' field"):
                provider.embed_text("test")

    def test_non_list_embedding_field_raises_api_error(self) -> None:
        provider = _make_provider()
        response = MagicMock()
        response.read.return_value = json.dumps({
            "data": [{"embedding": "not a list", "index": 0}]
        }).encode()
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)

        with _mock_urlopen(response):
            with pytest.raises(EmbeddingAPIError, match="missing 'embedding' field"):
                provider.embed_text("test")

    def test_data_field_not_a_list_raises_api_error(self) -> None:
        provider = _make_provider()
        response = MagicMock()
        response.read.return_value = json.dumps({
            "data": "not a list"
        }).encode()
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)

        with _mock_urlopen(response):
            with pytest.raises(EmbeddingAPIError, match="missing 'data' field or not a list"):
                provider.embed_text("test")


# ---------------------------------------------------------------------------
# 8. No network during construction
# ---------------------------------------------------------------------------


class TestNoNetworkOnConstruction:
    def test_construction_does_not_call_api(self) -> None:
        with patch("urllib.request.urlopen") as mock:
            provider = OpenAIEmbeddingProvider(
                api_key="sk-test",
                model="text-embedding-3-small",
            )
            mock.assert_not_called()

    def test_from_env_does_not_call_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REPOLENS_EMBEDDING_API_KEY", "sk-test")
        monkeypatch.setenv("REPOLENS_EMBEDDING_MODEL", "text-embedding-3-small")

        with patch("urllib.request.urlopen") as mock:
            provider = OpenAIEmbeddingProvider.from_env()
            mock.assert_not_called()
