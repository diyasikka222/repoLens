"""Refund API endpoints."""

from app.services.refunds import request_refund, approve_refund
from app.services.checkout import validate_checkout


def handle_refund_request(order_id: int, amount: float, reason: str) -> dict:
    from app.repositories.orders import get_order
    order = get_order(order_id)
    if order is None:
        return {"error": "order not found"}
    refund = request_refund(order, amount, reason)
    return {"refund_id": refund.refund_id, "status": refund.status}


def handle_refund_approval(refund_id: int) -> dict:
    from app.repositories.refunds import get_refund
    refund = get_refund(refund_id)
    if refund is None:
        return {"error": "refund not found"}
    approve_refund(refund)
    return {"refund_id": refund_id, "status": "approved"}
