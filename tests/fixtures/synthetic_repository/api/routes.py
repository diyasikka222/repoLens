"""HTTP API route registration."""

API_ROUTES = {
    "/health": "status",
}


def register_routes(app) -> None:
    for route, handler in API_ROUTES.items():
        app.get(route)(handler)
