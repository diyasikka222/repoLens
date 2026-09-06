"""Checkout validation rules."""

from app.models.order import Order


def validate_order(order: Order) -> bool:
    return order.total > 0


def validate_user_can_checkout(user_id: int) -> bool:
    return user_id > 0
