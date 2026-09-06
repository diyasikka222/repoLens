"""Payment processing service."""

from app.models.order import Order
from app.config.settings import TAX_RATE


def process_payment(order: Order) -> bool:
    total = order.calculate_total()
    total_with_tax = total * (1 + TAX_RATE)
    order.mark_paid()
    return True


def calculate_payment_total(order: Order) -> float:
    return order.calculate_total() * (1 + TAX_RATE)
