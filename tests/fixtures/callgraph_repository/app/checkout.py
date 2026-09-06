from .payments import charge_card, refund
from .validators import sanitize as clean
from .validators import validate
from . import models
import app.payments as payments


class CartController:
    def __init__(self):
        self.cart = models.Cart()

    def buy(self, card):
        order = models.Cart()
        self.cart.add(card)
        valid = validate(card)
        cleaned = clean(str(card))
        self.cart.checkout(order)
        charge_card(card)
        payments.charge_card(card)
        refund(order)
        bookings = globals().get("bookings")
        return bookings.make()