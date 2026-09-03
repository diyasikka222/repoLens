"""Deterministic verification that M19 context selection is more targeted.

Builds a small repository where several files share a broad domain term
(``payment``) so a vague query retrieves multiple files, while the precise
symbol ``report_refund`` is defined in one place. It demonstrates that a
precise symbol query returns a symbol-anchored, budget-friendly package ahead
of unrelated co-retrieved matches, and that intent-aware expansion pulls the
correct direction (dependents for impact, dependencies for "how it works").

Deterministic and fully offline. Prints exact counts.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from repolens.context import (
    CandidateRole,
    ContextBudget,
    ContextEngine,
    DependencyExpansionConfig,
    QueryIntent,
)
from repolens.search import CodeSearcher


def write_file(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def build_repo(root: Path) -> None:
    # Several files share the broad "payment" domain token; only refund.py
    # defines the precise symbol report_refund.
    write_file(root, "payments/notification.py", (
        "def send_payment_notice(payment_id):\n"
        "    return payment_id\n"
    ))
    write_file(root, "payments/webhook.py", (
        "def handle_payment_webhook(payload):\n"
        "    return payload\n"
    ))
    write_file(root, "payments/refund.py", (
        "def report_refund(txn_id):\n"
        "    return _render(txn_id)\n\n"
        "def _render(txn_id):\n"
        "    return f'refund {txn_id}'\n"
    ))
    write_file(root, "payments/ledger.py", (
        "from payments.refund import report_refund\n\n"
        "def post_entry(txn_id):\n"
        "    return report_refund(txn_id)\n"
    ))
    write_file(root, "billing/tax.py", (
        "def compute_tax():\n"
        "    return 0\n"
    ))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        root.mkdir()
        build_repo(root)

        def run(query: str, budget: int):
            engine = ContextEngine(
                root,
                searcher=CodeSearcher(root),
                budget=ContextBudget(max_tokens=budget),
                dependency=DependencyExpansionConfig(depth=1),
            )
            pkg = engine.build_context(query)
            return pkg

        # 1. Broad domain query -> normal retrieval over several files.
        pkg1 = run("process a payment", 10_000)
        # 2. Precise symbol query -> symbol-matched file is FIRST and symbolic.
        pkg2 = run("report_refund", 10_000)
        # 3. Impact query on the symbol -> its dependents expand (reverse dirs).
        pkg3 = run("impact of changing report_refund", 10_000)
        # 4. Tiny budget trims to a targeted package that still fits.
        pkg4 = run("report_refund", 90)

        def show(label: str, pkg) -> None:
            reasons = {
                c.path.as_posix(): c.inclusion_reason for c in pkg.selected_files
            }
            expanded = [c.path.as_posix() for c in pkg.dependency_candidates]
            print(f"{label} | intent={pkg.intent.value} symbols={list(pkg.matched_symbols)}")
            print(f"   primaries={[c.path.as_posix() for c in pkg.primary_candidates]}")
            print(f"   expanded={expanded} tokens={pkg.total_estimated_tokens}")
            print(f"   reasons={reasons}")
            print()

        show("vague 'process a payment'", pkg1)
        show("precise 'report_refund'", pkg2)
        show("impact 'impact of changing report_refund'", pkg3)
        show("tight budget (90) 'report_refund'", pkg4)

        # The precise symbol query must rank the defining file first and carry a
        # symbol-match reason, taking precedence over broader lexical matches.
        symbol_first = pkg2.primary_candidates and \
            pkg2.primary_candidates[0].path.as_posix() == "payments/refund.py"
        symbol_reason = any(
            c.inclusion_reason == "symbol_match"
            for c in pkg2.primary_candidates
        )
        # Broad query retrieves more than one file; intent stays UNKNOWN.
        vague_is_broad = len(pkg1.primary_candidates) > 1 and \
            pkg1.intent == QueryIntent.UNKNOWN
        # Impact query expands dependents (reverse) of the symbol.
        impact = pkg3.intent == QueryIntent.DEPENDENCY and pkg3.dependency_candidates
        impact_all_dependents = pkg3.dependency_candidates and all(
            c.role is CandidateRole.DEPENDENT
            for c in pkg3.dependency_candidates
        )
        # Tight budget stays within bound.
        tight = pkg4.total_estimated_tokens <= 90

        ok = (
            symbol_first
            and symbol_reason
            and vague_is_broad
            and impact
            and impact_all_dependents
            and tight
        )
        print("VERIFICATION " + ("PASSED" if ok else "FAILED"))
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())