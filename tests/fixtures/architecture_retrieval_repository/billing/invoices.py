from store.models import order


def create(user_id):
    return order.Order(user_id)