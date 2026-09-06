from app import settings

from app.settings import API_KEYS


def bootstrap():
    return settings.API_KEYS or API_KEYS