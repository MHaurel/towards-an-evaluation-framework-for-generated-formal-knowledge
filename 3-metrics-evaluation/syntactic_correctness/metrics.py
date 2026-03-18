"""
metrics.py — Per-program quality metric for syntactic-correctness detection.

The metric is the **mean per-clause syntax score**: the average of the
individual clause scores produced by the Prolog syntax checker.

    score = mean(clause_scores)   ∈ [0.0, 1.0]

A score of 1.0 means every clause in the KB is syntactically correct.
A score of 0.5 means the clauses are on average half-correct (e.g. half pass
the syntax check and half fail with a single error each).

Special cases
-------------
- ``error`` set / no clauses found: the code was empty or could not be parsed
  at all.  Score → ``SCORE_ERROR`` (default 0.0).

The constant ``SCORE_ERROR`` is defined at the top of this file and can be
adjusted without touching the computation logic.

Edit this file to change how the metric is computed.
"""

from syntactic_correctness_analyzer import SyntacticCorrectnessResult

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

#: Score assigned when analysis fails entirely (empty code, parse error, …).
SCORE_ERROR: float = 0.0


# ---------------------------------------------------------------------------
# Metric function
# ---------------------------------------------------------------------------


def syntactic_correctness_score(result: SyntacticCorrectnessResult) -> float:
    """Compute the syntactic correctness score for a single Prolog program.

    Parameters
    ----------
    result:
        The :class:`~syntactic_correctness_analyzer.SyntacticCorrectnessResult`
        returned by ``SyntacticCorrectnessAnalyzer(code).analyze()``.

    Returns
    -------
    float
        A value in [0.0, 1.0].  Higher is better (more clauses are correct).
    """

    return result.score
