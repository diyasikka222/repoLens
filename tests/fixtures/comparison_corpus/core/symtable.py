"""In-memory symbol table and name resolution helpers.

Maps identifier names to their runtime value. Grows slowly and is not used
for wiring up a project's dependency graph.
"""


def locate_binding(name: str) -> object:
    """Look up a single name in the global symbol table."""
    return None


def set_binding(name: str, value: object) -> None:
    """Store a value under a name for later retrieval."""
    return None