from store.repositories import orders
from store.services import checkout


def catalog(request):
    return checkout.run(request) and orders.find_all()