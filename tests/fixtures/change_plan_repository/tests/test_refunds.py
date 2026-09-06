"""Tests for refund service (partial - missing coverage for edge cases)."""

from app.services.refunds import request_refund, approve_refund, calculate_refund_total
from app.models.order import Order
from app.models.refund import Refund


def test_request_refund():
    order = Order(1, 1, 50.0)
    refund = request_refund(order, 10.0, "defective")
    assert refund.amount == 10.0


def test_approve_refund():
    order = Order(1, 1, 50.0)
    refund = Refund(1, order, 10.0, "defective")
    assert approve_refund(refund) is True


def test_calculate_refund_total_defective():
    order = Order(1, 1, 100.0)
    assert calculate_refund_total(order, "defective") == 100.0


# TODO: add tests for refund validation, edge cases, and notifications
