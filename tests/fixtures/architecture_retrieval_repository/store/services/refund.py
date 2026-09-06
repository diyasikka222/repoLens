from store.repositories import orders
from store.services import checkout


def refund(request):
    order_id = request.get("order_id")
    checkout.run([order_id])
    return orders.cancel(order_id)