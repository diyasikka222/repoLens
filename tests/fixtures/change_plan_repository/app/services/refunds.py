"""Refund service - handles refund requests against existing orders."""

from app.models.order import Order
from app.models.refund import Refund
from app.repositories.refunds import save_refund
from app.services.payments import process_payment
from app.services.checkout import validate_checkout


def request_refund(order: Order, amount: float, reason: str) -> Refund:
    refund = Refund(refund_id=0, order=order, amount=amount, reason=reason)
    save_refund(refund)
    return refund


def approve_refund(refund: Refund) -> bool:
    refund.approve()
    return True


def calculate_refund_total(order: Order, reason: str) -> float:
    base = order.calculate_total()
    if reason == "defective":
        return base
    return base * 0.5
