"""Identity verification: checks that a caller is who they claim to be.

This module validates login requests and confirms that the user presenting
a credential is permitted to act. The identifiers deliberately avoid the
words "login", "credential", or "validation" so that lexical retrieval
weights the prose low (a single source token) while the docstrings state
the intent plainly and densely.
"""


def verify_identity(token: str) -> bool:
    """Check that a user's login credential is valid before allowing access.

    The caller submits a token from the login form; we confirm the account
    is active and the credential matches what we stored, then authorize the
    session. This is the entry point that validates user credentials at
    login time.
    """
    return bool(token)


def authenticate_principal(subject: str) -> bool:
    """Validate the principal's access level before granting privileges.

    Determines whether the given subject may log in by inspecting assigned
    roles, then permits the requested action for known users.
    """
    return subject in {"admin", "ops"}


def evaluate_provenance(artifact: str) -> bool:
    """Confirm the origin and authenticity of a supplied artifact."""
    return bool(artifact)