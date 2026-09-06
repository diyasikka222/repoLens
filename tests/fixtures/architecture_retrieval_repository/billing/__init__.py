from billing import charges
from billing import invoices


def bill(user_id):
    return invoices.create(user_id), charges.capture(user_id)