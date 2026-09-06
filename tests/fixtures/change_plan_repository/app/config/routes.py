"""Route configuration."""

from app.config.settings import DATABASE_URL


ROUTES = {
    "/checkout": "app.api.checkout",
    "/refunds": "app.api.refunds",
    "/products": "app.api.products",
}


def get_route(path: str) -> str | None:
    return ROUTES.get(path)
