from .models import Order, tax


def charge_card(card):
    return True


def refund(order):
    return True


def process_payment(order: Order, card):
    total = tax(order.total())
    result = charge_card(card)
    return result