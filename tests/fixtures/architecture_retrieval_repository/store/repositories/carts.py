from store.models import cart


def find_cart(user_id):
    return cart.Cart(user_id)