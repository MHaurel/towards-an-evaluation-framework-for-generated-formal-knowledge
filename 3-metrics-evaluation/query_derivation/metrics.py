"""
metrics.py — Per-program query-derivation quality scores.

Two complementary scores:

``query_derivation_success``
    Did the execution query return ≥1 solution against the generated KB?

``query_derivation_accuracy``
    Among derived answers, does ``expected_output`` appear exactly?

Edit this file to change how the metric is computed from
:class:`~query_derivation_analyzer.QueryDerivationResult`.
"""

from query_derivation_analyzer import QueryDerivationResult

# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def query_derivation_success_score(result: QueryDerivationResult) -> float | None:
    """Return 1.0 / 0.0 / None for derivation success (see analyzer docstring)."""
    return result.success


def query_derivation_accuracy_score(result: QueryDerivationResult) -> float | None:
    """Return 1.0 / 0.0 / None for exact-match accuracy (see analyzer docstring)."""
    return result.accuracy
