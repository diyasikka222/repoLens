"""Refund repository."""

from app.models.order import Order
from app.models.refund import Refund


def get_refund(refund_id: int) -> Refund | None:
    order = Order(1, user_id=1, total=99.99)
    return Refund(refund_id, order, 10.0, "defective")


def save_refund(refund: Refund) -> None:
    pass


def list_refunds_for_order(order_id: int) -> list[Refund]:
    return []
