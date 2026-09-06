from app.services import users as service_users
from app.repositories import users as repository_users
from app.models import user
from app import services

import json


def list_users():
    return service_users