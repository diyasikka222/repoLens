from store.models import order
from store.models import cart


class Store:
    def catalog(self):
        return [order, cart]