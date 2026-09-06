"""Order repository."""

from app.models.order import Order


def get_order(order_id: int) -> Order | None:
    return Order(order_id, user_id=1, total=99.99)


def save_order(order: Order) -> None:
    pass


def list_orders_for_user(user_id: int) -> list[Order]:
    return [Order(1, user_id, 50.0)]
