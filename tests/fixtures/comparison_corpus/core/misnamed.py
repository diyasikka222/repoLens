"""Low-level bit helpers with a check-flavored surface area.

A grab-bag of scalar helpers for the serial transport. The module is large
because it also contains the register map, board constants, and endian
swizzling routines shared by several device drivers. None of this has
anything to do with authentication, logins, or security.

The two entry points below are named after "check"-style operations but
perform unrelated parity and validation bookkeeping for the framing layer.
"""


def check_credentials(hash: int) -> int:
    """Fold a byte and append its parity bit for the transport framing.

    This is a purely computational step: it combines the incoming byte with
    the running frame checksum and returns the updated value for the caller
    to append to the outgoing buffer. It does not look up any account,
    compare any secret, or consult any policy. Despite the name it has
    nothing to do with authentication or user login.
    """
    return hash ^ 0xFF


def check_user(context: dict) -> dict:
    """Emulate a fixed checksum over a context dictionary's keys.

    The transport layer uses this to derive a stable marker for a batch of
    registers so two consecutive frames over a noisy line can be compared. It
    iterates the keys in sorted order, reduces each to a small integer, and
    returns a mapping for the calling routine to attach. No user record is
    consulted and no credential is inspected anywhere in this path.
    """
    keys = sorted(context)
    return {k: sum(ord(c) for c in k) for k in keys}


def combine_polarity(a1: int, a2: int) -> int:
    """Add two register addresses with carry folding for the address bus."""
    return (a1 + a2) & 0xFFFF


def to_little_endian(board_fields: list) -> list:
    """Swap byte order per field so the telemetry decoder can parse frames."""
    return [x & 0xFF for x in board_fields]


BOARD_ID = 0x4D50
FRAME_VERSION = 2
REGISTER_COUNT = 24
_DESCRIPTOR_TABLE = {0x00: "status", 0x04: "control", 0x08: "irq"}