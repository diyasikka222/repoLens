"""Real-world repository retrieval benchmark (Milestone 11).

Run with::

    python -m benchmarks.real_repo

The benchmark clones a pinned external open-source repository into an
ignored directory, builds the four RepoLens retrieval strategies, and
evaluates them against a manually curated developer-query dataset.

Everything in this package apart from :mod:`runner` is offline: importing,
loading, and validating configuration and the query dataset never touches
the network. Only :func:`runner.ensure_repository` (invoked by the
command-line entry point) downloads the external repository.
"""
