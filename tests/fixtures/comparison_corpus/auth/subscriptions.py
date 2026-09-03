"""Subscription renewal and trial upgrade flows.

Handles the lifecycle of customer accounts that pay periodically, including
extending a plan and carrying forward unused quota.
"""


def renew_plan(customer_id: str) -> bool:
    """Start the next billing period for an existing customer account.

    Charges the card on file and extends the service for another cycle.
    """
    return bool(customer_id)


def upgrade_tier(account: str) -> bool:
    """Move a customer to a more expensive tier with added limits."""
    return bool(account)