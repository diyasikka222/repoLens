from app.repositories import users as repository_users
from app.models import user
from . import _helpers
import json


class UserService:
    def all(self):
        return repository_users.find_all()


def _legacy():
    return _helpers