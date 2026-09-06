"""Checkout service - core business logic for processing checkouts."""

from app.models.order import Order
from app.repositories.orders import save_order
from app.services.payments import process_payment


def checkout(user_id: int, cart_total: float) -> Order:
    order = Order(order_id=0, user_id=user_id, total=cart_total)
    save_order(order)
    process_payment(order)
    return order


def calculate_checkout_total(items: list[dict]) -> float:
    return sum(item.get("price", 0) for item in items)


def validate_checkout(user_id: int, total: float) -> bool:
    return user_id > 0 and total >= 0
