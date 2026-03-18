"""
metrics.py - Per-program quality metric for unused-terms detection.

The metric is the **clause utilization score**: the fraction of user-defined
clause instances that are *exercised* during the execution of the standard
query ``?- diagnosis(X).``

    score = |used_clauses| / |all_clauses|   ∈ [0.0, 1.0]

A score of 1.0 means every clause/fact defined in the KB was visited in at
least one proof branch (perfect utilization).  A score of 0.5 means half of
the clauses are dead code.

Unlike the old predicate-level metric, this score detects unused *individual
facts* such as ``symptom(test1)`` even when the predicate ``symptom/1`` itself
is exercised by other facts.

Special cases
-------------
- ``method == 'query_failed'``: the program could not answer the query at all
  (e.g. wrong predicate arity for diagnosis/N, infinite recursion, etc.).
  Score → ``SCORE_QUERY_FAILED`` (default 0.0).  These programs are "badly
  formed" and receive the worst possible score.
- ``method == 'none'`` / error: the code was empty or both analysers crashed.
  Score → ``SCORE_ERROR`` (default 0.0).
- ``method == 'static'``: the KB could not be loaded by SWI-Prolog so a
  regex-based fallback was used.  The score is still computed from the static
  result but is penalised by ``STATIC_PENALTY`` (default 0.1) to reflect
  lower confidence.

All three constants are defined at the top of this file so they can be
adjusted without touching the computation logic.

Edit this file to change how the metric is computed.
"""

from unused_terms_analyzer import UnusedTermsResult

# ---------------------------------------------------------------------------
# Tuneable constants – edit these to change the metric behaviour
# ---------------------------------------------------------------------------

#: Score assigned when the standard query returns no solutions.
SCORE_QUERY_FAILED: float = 0.0

#: Score assigned when both dynamic and static analysis fail entirely.
SCORE_ERROR: float = 0.0

#: Penalty subtracted from the raw utilisation ratio when the result comes
#: from static (regex) analysis rather than dynamic (SWI-Prolog) tracing.
#: Set to 0.0 to treat static results the same as dynamic ones.
STATIC_PENALTY: float = 0.1


# ---------------------------------------------------------------------------
# Metric function
# ---------------------------------------------------------------------------


def predicate_utilization_score(result: UnusedTermsResult) -> float:
    """
    Compute the clause utilization score for a single Prolog program.

    Uses clause-level granularity (``all_clauses`` / ``used_clauses``) when
    available (dynamic analysis), and falls back to predicate-level when only
    static analysis was possible.

    Parameters
    ----------
    result:
        The :class:`~unused_terms_analyzer.UnusedTermsResult` returned by
        ``UnusedTermsAnalyzer.analyze()``.

    Returns
    -------
    float
        A value in [0.0, 1.0].  Higher is better (more clauses used).
    """
    if result.method in ("none",) or result.error and result.method == "none":
        return SCORE_ERROR

    if result.method == "query_failed":
        return SCORE_QUERY_FAILED

    # Static fallback: clause-level data unavailable, use predicate-level.
    if result.method == "static":
        n_all = len(result.all_predicates)
        if n_all == 0:
            return 1.0
        raw_score = len(result.used_predicates) / n_all
        return round(max(0.0, raw_score - STATIC_PENALTY), 4)

    # Dynamic: use clause-level sets.
    n_all = len(result.all_clauses)
    if n_all == 0:
        # Edge case: program has no user-defined clauses at all.
        return 1.0

    raw_score = len(result.used_clauses) / n_all
    return round(raw_score, 4)
