"""Entry point for ``python -m benchmarks.real_repo``."""

from __future__ import annotations

import sys

from benchmarks.real_repo.runner import main

if __name__ == "__main__":
    sys.exit(main())
