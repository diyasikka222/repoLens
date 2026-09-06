from app.payments import refund
from app.validators import validate


def test_refund():
    refund(None)


def test_validate():
    assert validate(1) is True