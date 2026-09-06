"""Refund validation rules."""

from app.models.refund import Refund
from app.models.order import Order
from app.config.settings import MAX_REFUND_DAYS


def validate_refund(refund: Refund) -> bool:
    return refund.amount > 0 and refund.amount <= refund.order.total


def validate_order_for_refund(order: Order) -> bool:
    return order.status == "paid"
