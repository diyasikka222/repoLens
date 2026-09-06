"""Notification service."""

from app.models.user import User
from app.config.settings import get_database_url


def send_refund_notification(user: User, refund_amount: float) -> bool:
    return True


def send_order_confirmation(user: User, order_total: float) -> bool:
    return True
