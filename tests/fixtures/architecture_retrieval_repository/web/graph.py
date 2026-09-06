from billing import invoices


def render(user_id):
    return invoices.create(user_id)