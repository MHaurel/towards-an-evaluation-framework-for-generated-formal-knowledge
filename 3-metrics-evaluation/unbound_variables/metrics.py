"""
metrics.py — Per-program quality metric for unbound-variables detection.

The metric is the **clause cleanliness score**: the fraction of user-defined
clauses that contain *no* singleton variables.

    score = |clauses_without_singletons| / |total_clauses|   ∈ [0.0, 1.0]

A score of 1.0 means every clause in the KB is free of singleton (unused
unbound) variables — perfect.  A score of 0.5 means half of the clauses
contain at least one unused variable.

Special cases
-------------
- ``error`` set / ``method == 'static'`` with no clauses: the code was empty
  or could not be parsed at all.  Score → ``SCORE_ERROR`` (default 0.0).
- Zero clauses parsed (no Prolog content found): treated the same as an error.

The constant ``SCORE_ERROR`` is defined at the top of this file and can be
adjusted without touching the computation logic.

Edit this file to change how the metric is computed.
"""

from unbound_variables_analyzer import UnboundVariablesResult

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

#: Score assigned when analysis fails entirely (empty code, parse error, …).
SCORE_ERROR: float = 0.0


# ---------------------------------------------------------------------------
# Metric function
# ---------------------------------------------------------------------------


def clause_cleanliness_score(result: UnboundVariablesResult) -> float:
    """Compute the clause cleanliness score for a single Prolog program.

    Parameters
    ----------
    result:
        The :class:`~unbound_variables_analyzer.UnboundVariablesResult`
        returned by ``UnboundVariablesAnalyzer(code).analyze()``.

    Returns
    -------
    float
        A value in [0.0, 1.0].  Higher is better (fewer singleton variables).
    """
    # Error / empty-code path
    if result.error or result.total_clauses == 0:
        return SCORE_ERROR

    clean_clauses = result.total_clauses - len(result.clauses_with_singletons)
    return round(clean_clauses / result.total_clauses, 4)
