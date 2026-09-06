from store.services import checkout


def test_checkout_flow():
    assert checkout.run([1]) == {"total": 0, "items": [1]}