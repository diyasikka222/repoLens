"""Deterministic approximate token estimation.

RepoLens intentionally avoids an external tokenizer dependency for context
budgeting. Instead it uses a documented, deterministic approximation:

    estimated_tokens(text) = max(1, ceil(len(text) / 4))

i.e. roughly four characters per token. This is a coarse heuristic (an LLM
tokenizer typically reports a different, model-specific count) and is used
only to *budget* context size, never to substitute for the model's own
tokenization.

The estimate is:
- deterministic (pure function of the text),
- monotonic (longer text never estimates fewer tokens),
- strictly positive for any non-empty input,
- cheap to compute even on large files.

Distinguish clearly:
- :func:`estimate_tokens` approximates tokens for budgeting;
- the actual token count as counted by an LLM tokenizer may differ and is
  never measured here.
"""

from __future__ import annotations

import math

CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Return the approximate number of tokens for ``text``.

    ``max(1, ceil(len(text) / 4))``, so empty text estimates 1 token and the
    estimate grows monotonically with length.
    """
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))
