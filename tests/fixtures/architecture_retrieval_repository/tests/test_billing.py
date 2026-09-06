from billing.invoices import create


def test_invoices():
    assert create(1) is not None