"""Tests for payment processing."""

from app.services.payments import process_payment, calculate_payment_total
from app.models.order import Order


def test_process_payment():
    order = Order(1, 1, 100.0)
    assert process_payment(order) is True
    assert order.status == "paid"


def test_calculate_payment_total():
    order = Order(1, 1, 100.0)
    total = calculate_payment_total(order)
    assert total > 100.0
