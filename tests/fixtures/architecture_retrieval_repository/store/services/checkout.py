from store.repositories import orders
from store.models import order


def run(basket):
    if not basket:
        return []
    items = orders.store(basket)
    total = sum(order.line_item for _ in items for order in [order])
    return {"total": total, "items": items}