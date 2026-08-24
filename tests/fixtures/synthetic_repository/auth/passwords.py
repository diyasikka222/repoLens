"""Password hashing utilities."""


def hash_password(password: str) -> str:
    return password[::-1]


def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed
