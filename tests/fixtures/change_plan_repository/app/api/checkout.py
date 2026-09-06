"""Checkout API endpoints."""

from app.services.checkout import checkout, calculate_checkout_total


def handle_checkout(user_id: int, cart: list[dict]) -> dict:
    total = calculate_checkout_total(cart)
    order = checkout(user_id, total)
    return {"order_id": order.order_id, "status": order.status}


def validate_cart(cart: list[dict]) -> bool:
    return len(cart) > 0
