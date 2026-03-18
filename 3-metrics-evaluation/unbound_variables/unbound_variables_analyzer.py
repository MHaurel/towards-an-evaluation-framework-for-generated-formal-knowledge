"""
unbound_variables_analyzer.py — Detect singleton (unused unbound) variables in
Prolog programs.

Background
----------
In Prolog a *singleton variable* is a variable that appears **exactly once**
inside a single clause (head + body combined).  Because Prolog variables unify
across their clause, a variable that only appears once can never participate in
any unification or data-flow — it is semantically meaningless and is almost
always either a typo or dead code.

SWI-Prolog itself emits ``Singleton variables: [X, Z]`` warnings at load time
for exactly this reason.

Intentionally-ignored variables are spelled with a leading underscore (``_X``,
``_Unused``, etc.) or as the anonymous variable ``_``.  These are *not* flagged.

Approach
--------
Pure **static analysis** — no SWI-Prolog subprocess required.

For each clause the algorithm:

1. Strips comments (``% …`` and ``/* … */``).
2. Masks quoted strings (``"…"``) and quoted atoms (``'…'``) so that
   uppercase letters inside them are not mistaken for variables.
3. Collects every token matching ``[A-Z][A-Za-z0-9_]*`` (a Prolog variable).
4. Ignores the anonymous variable ``_`` and any name starting with ``_``.
5. Counts occurrences of each variable name in the clause.
6. Variables with count == 1 are singletons.

Clauses are split on ``.`` that is followed by whitespace or end-of-string,
which handles the most common clause terminator without confusing ``=..``,
floating-point literals, or qualified module calls (``module:pred``).
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClauseInfo:
    """Singleton-variable information for a single Prolog clause or fact."""

    clause_number: int
    """1-indexed position of this clause in the source file."""

    clause_text: str
    """Original (untrimmed) text of the clause, without the trailing ``'.'``."""

    singleton_variables: List[str]
    """Sorted list of variable names that appear exactly once in this clause."""

    @property
    def has_singletons(self) -> bool:
        return bool(self.singleton_variables)

    def __repr__(self) -> str:
        if self.singleton_variables:
            return (
                f"ClauseInfo(#{self.clause_number}, "
                f"singletons={self.singleton_variables})"
            )
        return f"ClauseInfo(#{self.clause_number}, clean)"


@dataclass
class UnboundVariablesResult:
    """Result of singleton-variable analysis for a single Prolog program."""

    all_clauses: List[ClauseInfo] = field(default_factory=list)
    """One :class:`ClauseInfo` per clause/fact found in the source."""

    clauses_with_singletons: List[ClauseInfo] = field(default_factory=list)
    """Subset of ``all_clauses`` that contain at least one singleton variable."""

    total_clauses: int = 0
    """Total number of clauses/facts parsed."""

    total_singletons: int = 0
    """Total count of distinct singleton-variable occurrences across all clauses."""

    has_singletons: bool = False
    """``True`` if at least one singleton variable was found anywhere."""

    singleton_ratio: float = 0.0
    """Fraction of clauses that contain at least one singleton (0.0–1.0)."""

    method: str = "static"
    """Always ``'static'`` for this analyser; kept for API consistency."""

    error: str = ""
    """Non-empty if parsing failed or the input was empty."""

    def __repr__(self) -> str:
        if self.error:
            return (
                f"UnboundVariablesResult(has_singletons={self.has_singletons}, "
                f"method={self.method!r}, error={self.error!r})"
            )
        return (
            f"UnboundVariablesResult("
            f"has_singletons={self.has_singletons}, "
            f"total_clauses={self.total_clauses}, "
            f"total_singletons={self.total_singletons}, "
            f"singleton_ratio={self.singleton_ratio:.2f}, "
            f"method={self.method!r})"
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Matches a Prolog variable: starts with an uppercase letter.
# (Variables starting with _ are intentionally anonymous by convention and
#  are excluded during filtering, not here.)
_VAR_RE = re.compile(r"\b([A-Z_][A-Za-z0-9_]*)\b")

# Matches a quoted atom '...' or a double-quoted string "..."
# We replace their contents with spaces so inner uppercase letters are ignored.
_QUOTED_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")

# Matches a line comment % ... until end of line
_LINE_COMMENT_RE = re.compile(r"%[^\n]*")

# Matches a block comment /* ... */ (non-greedy, dotall)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# Clause terminator: a literal '.' that is:
#   • NOT preceded by '.' or '=' — avoids splitting inside '=..' (univ operator)
#   • Followed by whitespace or end-of-string — avoids splitting on floats (3.14)
#     or module qualifiers (lists:member)
_CLAUSE_SEP_RE = re.compile(r"(?<![.=])\.\s*(?=\s|$)")


def _strip_comments(code: str) -> str:
    """Remove ``% …`` line comments and ``/* … */`` block comments."""
    code = _BLOCK_COMMENT_RE.sub(" ", code)
    code = _LINE_COMMENT_RE.sub("", code)
    return code


def _mask_quoted(text: str) -> str:
    """Replace the *contents* of quoted atoms/strings with underscores.

    This prevents uppercase letters inside ``'Atom'`` or ``"String"`` from
    being mis-identified as Prolog variables.
    """
    def _blank(m: re.Match) -> str:
        # Keep the delimiters, replace contents with underscores of same length
        inner = m.group(0)
        return inner[0] + ("_" * (len(inner) - 2)) + inner[-1]

    return _QUOTED_RE.sub(_blank, text)


def _extract_variables(clause_text: str) -> List[str]:
    """Return a list of every Prolog variable token in *clause_text*.

    Variables are tokens matching ``[A-Z][A-Za-z0-9_]*``.  Anonymous variables
    (``_`` alone or ``_Prefixed``) are excluded.
    """
    masked = _mask_quoted(clause_text)
    raw = _VAR_RE.findall(masked)
    # Exclude _ alone and _Prefixed names (intentionally ignored in Prolog)
    return [v for v in raw if v != "_" and not v.startswith("_")]


def _singleton_variables(clause_text: str) -> List[str]:
    """Return sorted list of singleton variable names in *clause_text*."""
    vars_ = _extract_variables(clause_text)
    counts = Counter(vars_)
    return sorted(name for name, cnt in counts.items() if cnt == 1)


def _split_clauses(code: str) -> List[str]:
    """Split cleaned Prolog source into individual clause strings.

    The trailing ``.`` is consumed by the split; each returned string contains
    the clause text without its terminator.
    """
    parts = _CLAUSE_SEP_RE.split(code)
    # Drop empty or whitespace-only fragments
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Main analyser class
# ---------------------------------------------------------------------------

class UnboundVariablesAnalyzer:
    """Detect singleton (unused unbound) variables in a Prolog program.

    Usage::

        result = UnboundVariablesAnalyzer(code).analyze()

    Parameters
    ----------
    code:
        Prolog source code as a plain string.
    """

    def __init__(self, code: str) -> None:
        self.code = code

    def analyze(self) -> UnboundVariablesResult:
        """Run the analysis and return an :class:`UnboundVariablesResult`."""
        if not self.code or not self.code.strip():
            return UnboundVariablesResult(
                has_singletons=False,
                error="Empty code",
                method="static",
            )

        try:
            return self._analyze()
        except Exception as exc:  # pragma: no cover
            return UnboundVariablesResult(
                has_singletons=False,
                error=f"Analysis failed: {exc}",
                method="static",
            )

    def _analyze(self) -> UnboundVariablesResult:
        # 1. Strip comments
        clean = _strip_comments(self.code)

        # 2. Split into clauses
        raw_clauses = _split_clauses(clean)

        if not raw_clauses:
            return UnboundVariablesResult(
                has_singletons=False,
                error="No clauses found",
                method="static",
            )

        # 3. Analyse each clause
        all_clause_infos: List[ClauseInfo] = []
        for idx, clause_text in enumerate(raw_clauses, start=1):
            singletons = _singleton_variables(clause_text)
            info = ClauseInfo(
                clause_number=idx,
                clause_text=clause_text.strip(),
                singleton_variables=singletons,
            )
            all_clause_infos.append(info)

        # 4. Aggregate
        clauses_with_singletons = [c for c in all_clause_infos if c.has_singletons]
        total_singletons = sum(len(c.singleton_variables) for c in all_clause_infos)
        total_clauses = len(all_clause_infos)
        singleton_ratio = (
            len(clauses_with_singletons) / total_clauses if total_clauses > 0 else 0.0
        )

        return UnboundVariablesResult(
            all_clauses=all_clause_infos,
            clauses_with_singletons=clauses_with_singletons,
            total_clauses=total_clauses,
            total_singletons=total_singletons,
            has_singletons=total_singletons > 0,
            singleton_ratio=singleton_ratio,
            method="static",
        )
