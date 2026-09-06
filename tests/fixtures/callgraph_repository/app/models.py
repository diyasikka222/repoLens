class Order:
    def total(self):
        return 100


class Cart:
    def add(self, item):
        return True

    def checkout(self, order):
        return order.total()


def tax(amount):
    return amount * 0.08