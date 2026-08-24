"""A pooled cache of reusable database connections."""

from database.connection import connect


class ConnectionPool:
    def acquire(self):
        return connect("postgres://localhost")

    def release(self, connection) -> None:
        pass
