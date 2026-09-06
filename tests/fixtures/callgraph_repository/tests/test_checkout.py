from app import payments
from app.checkout import CartController
from app.payments import process_payment


def test_buy():
    c = CartController()
    c.buy("1234")
    payments.charge_card("1234")


def test_process():
    process_payment(None, "1234")