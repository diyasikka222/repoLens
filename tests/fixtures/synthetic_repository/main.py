"""Application entry point wiring modules together."""

from api.routes import register_routes
from auth.login import LoginService
from database.pool import ConnectionPool


def bootstrap():
    login = LoginService()
    pool = ConnectionPool()
    register_routes(login)
    return login, pool
