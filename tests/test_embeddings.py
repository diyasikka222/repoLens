"""Tests for the embedding provider abstraction and the offline provider."""

import math

import pytest

from repolens.embeddings import EmbeddingProvider, FakeEmbeddingProvider


@pytest.fixture()
def provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


# --- 1. Interface behavior ---------------------------------------------------


def test_fake_provider_satisfies_embedding_provider_protocol(
    provider: FakeEmbeddingProvider,
) -> None:
    assert isinstance(provider, EmbeddingProvider)


def test_embed_texts_preserves_order_and_matches_embed_text(
    provider: FakeEmbeddingProvider,
) -> None:
    texts = ["invoice calculation", "database connection", ""]

    batched = provider.embed_texts(texts)

    assert len(batched) == len(texts)
    assert batched == tuple(provider.embed_text(text) for text in texts)


def test_vectors_have_configured_length_and_unit_norm(
    provider: FakeEmbeddingProvider,
) -> None:
    vector = provider.embed_text("user authentication")

    assert len(vector) == provider.dimensions
    assert all(isinstance(component, float) for component in vector)
    assert math.sqrt(sum(component * component for component in vector)) == pytest.approx(1.0)


def test_custom_dimension_count_is_respected() -> None:
    provider = FakeEmbeddingProvider(dimensions=8)

    assert len(provider.embed_text("anything at all")) == 8

    with pytest.raises(ValueError):
        FakeEmbeddingProvider(dimensions=0)


# --- 2./3. Deterministic vectors ------------------------------------------------


def test_same_instance_produces_identical_vectors_for_repeated_calls(
    provider: FakeEmbeddingProvider,
) -> None:
    first = provider.embed_text("refund a card payment")
    second = provider.embed_text("refund a card payment")

    assert first == second


def test_separate_instances_produce_identical_vectors() -> None:
    left = FakeEmbeddingProvider().embed_text("process payment")
    right = FakeEmbeddingProvider().embed_text("process payment")

    assert left == right


# --- 4. Different texts differ ----------------------------------------------------


def test_different_texts_produce_different_vectors(
    provider: FakeEmbeddingProvider,
) -> None:
    invoice_vector = provider.embed_text("invoice calculation total")
    database_vector = provider.embed_text("database connection pool")

    assert invoice_vector != database_vector


def test_word_order_does_not_matter_bag_of_words(
    provider: FakeEmbeddingProvider,
) -> None:
    forward = provider.embed_text("card payment refund")
    backward = provider.embed_text("refund payment card")

    assert forward == backward


def test_empty_or_separator_only_text_yields_zero_vector(
    provider: FakeEmbeddingProvider,
) -> None:
    zeros = tuple(0.0 for _ in range(provider.dimensions))

    assert provider.embed_text("") == zeros
    assert provider.embed_text("   !!! --- ") == zeros
