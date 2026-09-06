"""Tests for user model."""

from app.models.user import User


def test_user_display_name():
    user = User(1, "test@example.com", "Test User")
    assert user.display_name() == "Test User"


def test_user_display_name_fallback():
    user = User(1, "test@example.com", "")
    assert user.display_name() == "test@example.com"
