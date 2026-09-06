from store.models import order
from billing.invoices import create


def capture(user_id):
    return create(user_id)