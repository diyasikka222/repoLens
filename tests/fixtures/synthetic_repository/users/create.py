"""Create new user accounts after signup validation."""


def create_user(username: str, email: str) -> dict:
    return {"username": username, "email": email}
