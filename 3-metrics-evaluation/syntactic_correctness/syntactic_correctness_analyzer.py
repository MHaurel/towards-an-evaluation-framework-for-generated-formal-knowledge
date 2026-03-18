"""
syntactic_correctness_analyzer.py — Evaluate the syntactic correctness of
Prolog programs on a per-clause basis.

Approach
--------
The source is split on ``.`` clause terminators.  Each non-empty clause is
evaluated independently and assigned a score in [0.0, 1.0].  The overall
result is the mean of per-clause scores.

The per-clause evaluation is delegated to :func:`_evaluate_clause`, which
contains a **placeholder** — replace it with the actual Prolog syntax checker
(e.g. a swipl subprocess call).
"""

from dataclasses import dataclass, field
from typing import List
from swiplParser.swiplParser import evaluate_prolog_syntax_per_clause


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SyntacticCorrectnessResult:
    """Result of syntactic-correctness analysis for a single Prolog program."""

    score: float


# ---------------------------------------------------------------------------
# Main analyser class
# ---------------------------------------------------------------------------

class SyntacticCorrectnessAnalyzer:
    """Evaluate syntactic correctness of a Prolog program per clause.

    Usage::

        result = SyntacticCorrectnessAnalyzer(code).analyze()

    Parameters
    ----------
    code:
        Prolog source code as a plain string.
    """

    def __init__(self, code: str) -> None:
        self.code = code

    def analyze(self) -> SyntacticCorrectnessResult:
        """Run the analysis and return a :class:`SyntacticCorrectnessResult`."""
        if not self.code or not self.code.strip():
            return SyntacticCorrectnessResult(
                mean_score=0.0,
                error="Empty code",
                method="swipl",
            )

        try:
            return self._analyze()
        except NotImplementedError:
            raise
        except Exception as exc:
            return SyntacticCorrectnessResult(
                mean_score=0.0,
                error=f"Analysis failed: {exc}",
                method="swipl",
            )

    def _analyze(self) -> SyntacticCorrectnessResult:
        # Split into clauses on '.' followed by whitespace or end-of-string.
        # Filter out empty fragments.
        if not self.code.strip():
            return SyntacticCorrectnessResult(
                mean_score=0.0,
                error="No clauses found",
                method="swipl",
            )

        score = evaluate_prolog_syntax_per_clause(self.code)


        return SyntacticCorrectnessResult(
            score=score
        )