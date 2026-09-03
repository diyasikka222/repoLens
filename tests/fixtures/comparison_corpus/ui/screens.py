"""User-facing screens and input formatters.

Renders pages and sanitizes the strings a user types into a form. Does not
enforce any security policy.
"""


def render_page(template: str) -> str:
    """Fill a template with data and return the rendered markup."""
    return template


def sanitise_input(raw: str) -> str:
    """Escape potentially hostile characters before reflecting user input."""
    return raw.replace("<", "&lt;")