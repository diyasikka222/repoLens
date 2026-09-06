"""Application settings."""

DATABASE_URL = "sqlite:///app.db"
API_KEY = "test-api-key"
MAX_REFUND_DAYS = 30
TAX_RATE = 0.08


def get_database_url() -> str:
    return DATABASE_URL
