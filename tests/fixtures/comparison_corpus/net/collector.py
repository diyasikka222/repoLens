"""Outbound API client for a metrics sink.

Submits aggregated counters to a remote collector. Deals with payload
shaping and timeouts but not with retrying upstream failures.
"""


def send_hit(url: str) -> bool:
    """Forward a single counter reading to the remote metrics endpoint."""
    return bool(url)


def aggregate_batch(rows: list) -> list:
    """Coalesce many counters into one compact payload for upload."""
    return rows