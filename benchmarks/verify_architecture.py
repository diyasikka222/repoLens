"""Deterministic verification for M23 repository-level architecture analysis.

Part 1 runs against a throwaway copy of the ``architecture_repository``
fixture and asserts the exact structure of
:class:`repolens.architecture.ArchitectureGraph`: file/module/package node
membership, containment, module- and package-level dependency edges, bounded
transitive traversal, deterministic serialization, and the incremental story
(one-file modification updates edges; file deletion removes the node and every
edge referencing it).

Part 2 runs against the real RepoLens repository: a cold architecture build
(scan + parse), then a warm build driven only by the incremental index (zero
files re-scanned — the architecture builder is a pure projection of existing
data). It asserts determinism across builds, prints real statistics, and
reports dependency-traversal latency.

Deterministic and fully offline. Uses temp caches; does not modify the
repository under analysis.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

from repolens.architecture import ArchitectureGraphBuilder, ArchitectureRelationship
from repolens.incremental_index import IncrementalIndexBuilder

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "architecture_repository"


def _ids(nodes) -> set[str]:
    return {node.id for node in nodes}


def build_fixture_copy(tmp: Path) -> Path:
    root = tmp / "repo"
    shutil.copytree(FIXTURE, root)
    return root


def part1(root: Path) -> None:
    graph = ArchitectureGraphBuilder(root).build()

    stats = graph.stats()
    assert stats.packages == 6, stats.as_dict()
    assert _ids(graph.packages()) == {
        ".",
        "app",
        "app/api",
        "app/models",
        "app/repositories",
        "app/services",
    }
    print("fixture graph")
    print(f"  stats: {stats.as_dict()}")
    print("  packages: OK (root + app + 4 nested)")

    # Containment.
    assert _ids(graph.files_in_package("app/services")) == {
        "app/services/__init__.py",
        "app/services/_helpers.py",
        "app/services/users.py",
    }
    assert _ids(graph.contains("app")) == {
        "app/__init__.py",
        "app/api",
        "app/models",
        "app/repositories",
        "app/services",
    }
    print("  containment (package -> file -> module): OK")

    # Module and package dependencies.
    assert _ids(graph.module_dependencies("app.api.routes")) == {
        "app.services",
        "app.services.users",
        "app.repositories.users",
        "app.models.user",
    }
    assert _ids(graph.package_dependencies("app/api")) == {
        "app/models",
        "app/repositories",
        "app/services",
    }
    print("  module + package dependency edges: OK")

    # Bounded transitive traversal.
    dependents = graph.transitive_dependents("app.models.user", max_depth=3)
    assert "app.services.users" in _ids(dependents)
    assert "app.models.user" not in _ids(dependents)
    assert graph.transitive_dependencies("app.api.routes", max_depth=0) == []
    print(f"  transitive dependents(app.models.user, 3): {len(dependents)} nodes")

    # Determinism.
    assert graph.serialize() == graph.serialize()
    print("  deterministic serialization: OK")

    # Incremental: add a module and import it.
    (root / "app" / "config.py").write_text(
        "SERVICE_NAME = 'accounts'\n", encoding="utf-8"
    )
    (root / "app" / "services" / "_helpers.py").write_text(
        "from app import config\n\ndef identity(value):\n    return value\n",
        encoding="utf-8",
    )
    modified = ArchitectureGraphBuilder(root).build()
    assert _ids(modified.module_dependencies("app.services._helpers")) == {"app.config"}
    assert "app" in _ids(modified.package_dependencies("app/services"))
    print("  one-file modification: new module + package edge appear: OK")

    # Incremental: delete a module; its node and all its edges must vanish.
    (root / "app" / "models" / "user.py").unlink()
    deleted = ArchitectureGraphBuilder(root).build()
    assert deleted.get_module_node("app.models.user") is None
    for edge in deleted.get_edges():
        assert edge.source.id != "app.models.user"
        assert edge.target.id != "app.models.user"
    assert _ids(deleted.module_dependencies("app.repositories.users")) == {"app.models"}
    print("  file deletion: node + stale edges removed: OK")


def part2(root: Path) -> None:
    cache_dir = Path(tempfile.mkdtemp(prefix="repolens-arch-real-"))
    index = IncrementalIndexBuilder(root, cache_dir=cache_dir / "index").build()
    cold = ArchitectureGraphBuilder(root, index=index).build()

    warm_index = IncrementalIndexBuilder(root, cache_dir=cache_dir / "index").build()
    warm = ArchitectureGraphBuilder(root, index=warm_index).build()

    stats = cold.stats()
    assert stats.nodes > 0 and stats.depends_on_edges > 0
    print("real repo: architecture graph")
    print(f"  stats: {stats.as_dict()}")

    # Warm builds must not re-parse or re-scan anything.
    assert warm_index.stats.files_parsed == 0, "warm index must not re-parse"
    print(f"  cold index parsed={index.stats.files_parsed} "
          f"warm index parsed={warm_index.stats.files_parsed}: OK")

    # Determinism across builds.
    assert cold.serialize() == warm.serialize()
    print("  deterministic across builds: OK")

    # Dependency-traversal latency on real modules with edges.
    # Dependency-traversal latency on a real module-level dependency edge.
    dep_edges = [
        e
        for e in cold.get_edges()
        if e.relationship is ArchitectureRelationship.DEPENDS_ON
    ]
    sample = next(iter({(e.source.id, e.target.id) for e in dep_edges}))
    start = time.perf_counter()
    result = cold.transitive_dependencies(sample[0], max_depth=4)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert sample[1] in _ids(result)
    print(f"  traversal: transitive_dependencies({sample[0]}, 4) "
          f"-> {len(result)} nodes in {elapsed_ms:.1f}ms")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        print("PART 1 - synthetic architecture repository")
        part1(build_fixture_copy(Path(tmp)))

    print("\nPART 2 - real repository (RepoLens itself)")
    part2(REPO_ROOT)

    print("\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())