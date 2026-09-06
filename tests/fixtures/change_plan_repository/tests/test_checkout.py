"""Tests for checkout service."""

from app.services.checkout import checkout, calculate_checkout_total, validate_checkout


def test_checkout_creates_order():
    order = checkout(user_id=1, cart_total=50.0)
    assert order.status == "paid"


def test_calculate_checkout_total():
    items = [{"price": 10.0}, {"price": 20.0}]
    assert calculate_checkout_total(items) == 30.0


def test_validate_checkout():
    assert validate_checkout(1, 10.0) is True
    assert validate_checkout(0, 10.0) is False
