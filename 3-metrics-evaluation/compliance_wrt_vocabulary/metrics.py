"""
metrics.py — Per-program type-compliance metric for the compliance_wrt_ontology
experiment.

The metric is the **type compliance score**: a single float in [0.0, 1.0] that
aggregates three sub-dimensions of compliance:

1. **Name compliance** — what fraction of clause heads use predicate names that
   appear in the reference ontology?
2. **Arity compliance** — of those known predicates, what fraction use the
   correct number of arguments?
3. **Type compliance** — for predicates with the correct arity, how semantically
   close are the actual argument values to their expected type labels on average?

These three sub-scores are combined into a single composite score.  The default
weighting is equal (1/3 each), but this can be overridden by passing a custom
``weights`` tuple.

Score interpretation
--------------------
* **1.0** — every clause uses a known predicate with the correct arity and
  semantically appropriate arguments.
* **0.0** — the program uses no recognised predicates, has systematic arity
  mismatches, or all argument embeddings are orthogonal to their expected types.

Not-evaluable cases
-------------------
The metric returns ``None`` when:

* The code was empty or no clauses could be parsed
  (``method in ("empty_code", "no_clauses")``).
* An unexpected error occurred during analysis (``method == "error"``).

These are represented by Python ``None`` so that pandas treats them as ``NaN``
when the result is stored in a DataFrame column, allowing callers to exclude
them from aggregations.

Sub-scores
----------
You can retrieve the individual sub-scores with
:func:`type_compliance_sub_scores` if you need finer-grained information for
analysis or debugging.

Edit the ``WEIGHTS`` constant in this file to change the default weighting
without touching any other code.
"""

from __future__ import annotations

from typing import Optional

from type_compliance_analyzer import TypeComplianceResult, METHOD_SEMANTIC

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

#: Default equal weighting for (name, arity, type) sub-scores.
#: Must sum to 1.0.  Edit here to change the composite score formula.
WEIGHTS: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)

#: Symbolic name for the not-evaluable return value.
SCORE_NOT_EVALUABLE: None = None


# ---------------------------------------------------------------------------
# Sub-score helper
# ---------------------------------------------------------------------------


def type_compliance_sub_scores(
    result: TypeComplianceResult,
) -> Optional[dict[str, Optional[float]]]:
    """Return the three individual sub-scores for a single analysis result.

    Parameters
    ----------
    result:
        The :class:`~type_compliance_analyzer.TypeComplianceResult` returned
        by ``TypeComplianceAnalyzer(...).analyze()``.

    Returns
    -------
    dict or None
        A dictionary with keys ``"name"``, ``"arity"``, and ``"type"``, or
        ``None`` when the result is not evaluable (empty/error).

        * ``"name"`` — fraction of clause heads with a known predicate name.
        * ``"arity"`` — fraction of clause heads with correct arity
          (counting only clauses whose name was found).
        * ``"type"`` — mean cosine similarity of arguments to their expected
          type labels across all arity-correct clauses.  May be ``None`` if
          no arity-correct clauses exist.
    """
    if result.method != METHOD_SEMANTIC or result.total_clauses == 0:
        return None

    name_score = round(result.name_compliance_score, 4)
    arity_score = round(result.arity_compliance_score, 4)
    type_score = result.mean_type_compliance_score
    if type_score is not None:
        type_score = round(type_score, 4)

    return {
        "name": name_score,
        "arity": arity_score,
        "type": type_score,
    }


# ---------------------------------------------------------------------------
# Composite metric function
# ---------------------------------------------------------------------------


def type_compliance_score(
    result: TypeComplianceResult,
    weights: tuple[float, float, float] = WEIGHTS,
) -> Optional[float]:
    """Compute the composite type-compliance score for a single Prolog program.

    The score is the weighted average of three sub-scores:

    * **Name compliance** — fraction of clause heads using ontology-known
      predicate names.
    * **Arity compliance** — fraction of clause heads with the correct number
      of arguments for their predicate (includes only known predicates).
    * **Type compliance** — mean cosine similarity between actual argument
      values and their expected semantic type labels (includes only known,
      arity-correct predicates).

    When no arity-correct clauses exist (the type sub-score is undefined),
    the composite is computed using only the name and arity sub-scores,
    weighted proportionally to their original weights.

    Parameters
    ----------
    result:
        The :class:`~type_compliance_analyzer.TypeComplianceResult` returned
        by ``TypeComplianceAnalyzer(...).analyze()``.
    weights:
        A 3-tuple ``(w_name, w_arity, w_type)`` summing to 1.0.  Defaults to
        equal weighting ``(1/3, 1/3, 1/3)``.

    Returns
    -------
    float or None
        * A value in ``[0.0, 1.0]``, rounded to 4 decimal places.  Higher is
          better (greater ontological compliance).
        * ``None`` if the program could not be evaluated (empty code, parse
          failure, unexpected error).  Corresponds to ``NaN`` in pandas.
    """
    sub = type_compliance_sub_scores(result)
    if sub is None:
        return SCORE_NOT_EVALUABLE

    w_name, w_arity, w_type = weights
    name_s: float = sub["name"] or 0.0    # always a float when sub is not None
    arity_s: float = sub["arity"] or 0.0  # always a float when sub is not None
    type_s: Optional[float] = sub["type"]  # may be None

    if type_s is None:
        # No arity-correct clauses → type sub-score is undefined.
        # Renormalise weights over the two available sub-scores.
        total_w = w_name + w_arity
        if total_w == 0.0:
            return SCORE_NOT_EVALUABLE
        composite = (w_name * name_s + w_arity * arity_s) / total_w
    else:
        composite = w_name * name_s + w_arity * arity_s + w_type * type_s

    return round(composite, 4)
