"""Product repository."""

from app.models.product import Product


def get_product(product_id: int) -> Product | None:
    return Product(product_id, "Widget", 19.99)


def list_products() -> list[Product]:
    return [Product(1, "Widget", 19.99), Product(2, "Gadget", 29.99)]
