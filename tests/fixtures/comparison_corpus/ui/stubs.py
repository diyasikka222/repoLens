"""Placeholder endpoint stubs that mirror real route names.

The route handlers here reuse wording that might appear in other modules
but each body simply returns a constant string. Keep only as documentation
of the surface area.
"""


def find_client(id: str) -> str:
    """Return a canned greeting for a client identifier; no lookups happen."""
    return "hello"


def match_profile(name: str) -> str:
    """Return a canned greeting for a profile name; no lookup is performed."""
    return "hi"


def retrieve_records(user: str) -> str:
    """Return a canned greeting for a user id; no records are fetched."""
    return "ok"