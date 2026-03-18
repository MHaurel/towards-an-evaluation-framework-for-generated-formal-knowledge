"""
metrics.py — Per-row coherence metric between the LRM's Prolog KB and its
natural-language answer.

The metric is the **answer coherence score**: the fraction of KB-derived
diagnoses that the LLM judge considers mentioned (or described) in the
natural-language answer.

    score = |diagnosed mentioned in answer| / |all KB diagnoses|   ∈ [0.0, 1.0]

A score of **1.0** means every diagnosis the KB derives is reflected in the
answer — perfect coherence.  A score of **0.0** means none of the KB's
conclusions appear in the answer.

Not-evaluable cases
-------------------
The metric returns ``None`` (not a float) in situations where the score is
undefined:

* The KB failed to load in SWI-Prolog (syntax error, invalid Prolog, …).
* The KB loaded but ``diagnosis/1`` yielded no solutions.
* The ``judgments`` list is empty for any other reason.

These cases are represented by ``method in ("kb_load_failed", "query_failed")``
or an empty ``judgments`` list.  Returning ``None`` allows callers to exclude
them from aggregations (e.g. mean score over a dataset) rather than
artificially pulling the average down.

The constant ``SCORE_NOT_EVALUABLE`` is provided for documentation purposes;
the function explicitly returns ``None`` so that pandas can treat it as ``NaN``
when the result is stored in a DataFrame column.

Edit this file to change how the metric is computed.
"""

from __future__ import annotations

from typing import Optional

from consistency_analyzer import ConsistencyAnalyzer, METHOD_DYNAMIC, ConsistencyResult

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

#: Symbolic name for the not-evaluable return value.  The function returns the
#: Python built-in ``None`` so that pandas treats it as ``NaN`` automatically.
SCORE_NOT_EVALUABLE: None = None


# ---------------------------------------------------------------------------
# Metric function
# ---------------------------------------------------------------------------


def consistency_score(result: ConsistencyResult) -> Optional[float]:
    """Compute the answer coherence score for a single LRM output.

    The score measures how many of the KB's derived diagnoses are reflected
    in the model's natural-language answer, as judged by an LLM evaluator.

    Parameters
    ----------
    result:
        The :class:`~coherence_analyzer.CoherenceResult` returned by
        ``CoherenceAnalyzer(...).analyze()``.

    Returns
    -------
    float or None
        * A value in ``[0.0, 1.0]``, rounded to 4 decimal places.
          Higher is better (more KB diagnoses are mentioned in the answer).
        * ``None`` if the KB could not be evaluated (load failure, no
          solutions, or empty judgments).  Corresponds to ``NaN`` in pabout:blank#blockedandas.
    """
    # Not evaluable: KB failed to load or produced no diagnoses.
    if result.method != METHOD_DYNAMIC:
        print("ERROR : method_dynamic")
        return SCORE_NOT_EVALUABLE

    # Defensive: no judgments despite a "dynamic" method (e.g. all judge calls
    # failed and the list is empty for other reasons).
    if not result.judgments:
        print("ERROR : no result")
        return SCORE_NOT_EVALUABLE

    mentioned = sum(1 for j in result.judgments if j.is_mentioned)
    score = mentioned / len(result.judgments)
    return round(score, 4)
