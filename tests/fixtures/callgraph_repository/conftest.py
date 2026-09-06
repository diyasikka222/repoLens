# The callgraph_repository fixture ships intentionally test-named modules so
# the call graph and impact analysis can exercise test-file discovery. Those
# are fixture *sources*, not tests to collect, so ignore this nested tests dir.
collect_ignore = ["tests"]