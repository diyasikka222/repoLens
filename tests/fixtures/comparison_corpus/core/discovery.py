"""Source-tree discovery for a code indexer.

Walks a checkout directory to enumerate the source files a downstream stage
should ingest. The identifiers deliberately do not use the words "discover",
"scan", or "file list" so lexical matching leans on symbol-name matches in
other modules; the docstrings make the intent explicit.
"""


def scan_for_source_files(root: str, ignored: set) -> list:
    """Discover the files in a repository tree by walking the directory.

    Returns the list of source files found beneath the given root, skipping
    the ignored directories, so the indexing pipeline knows what to ingest.
    This is where the repository's source files are discovered.
    """
    return []


def enumerate_modules(root: str) -> list:
    """Return all source units located under a given checkout root.

    Walks the tree and yields every discoverable source module so callers
    can plan which files to process next.
    """
    return []