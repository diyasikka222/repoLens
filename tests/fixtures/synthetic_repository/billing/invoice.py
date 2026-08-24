"""Invoice calculation for billed orders."""


class InvoiceCalculator:
    def calculate_total(self, items) -> int:
        return sum(items)


def create_invoice(order_id: str):
    return order_id
