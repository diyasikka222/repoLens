# The architecture_retrieval_repository fixture intentionally ships
# test-named modules inside a nested ``tests`` package (so the architecture
# graph can exercise test-file discovery and a tests subsystem). They are
# fixture *sources*, not tests to collect, so ignore that nested directory.
#
# Living here (instead of inside the fixture) keeps the fixture directory off
# pytest's sys.path, so it can never shadow the repository's own ``tests``
# package during collection.
collect_ignore = [
    "architecture_retrieval_repository/tests",
    "change_plan_repository/tests",
]