"""
query_derivation_analyzer.py — Run a KB query in SWI-Prolog and score derivation.

Loads a Datalog/Prolog program, executes an ``execution_query``, and records
whether any answers were derived and whether ``expected_output`` appears among
them.

Scores (exposed via :mod:`metrics`)
------------------------------------
``query_derivation_success``
  * ``1.0`` — query returned at least one answer
  * ``0.0`` — query ran cleanly but yielded zero solutions
  * ``None`` — not evaluable (missing code/query, load/query failure, timeout)

``query_derivation_accuracy``
  * ``1.0`` — at least one derived answer equals ``expected_output`` exactly
  * ``0.0`` — query returned answer(s), but none match exactly
  * ``None`` — not evaluable (no derivation, missing fields, load/query
    failure, timeout)

For model generations where program and query share a single ``<think>`` block
(Cogito / SFT format), use :func:`split_program_and_query` to separate them
(last clause = query, preceding clauses = program).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any


@dataclass
class QueryDerivationResult:
    """Result of running a query against a generated KB."""

    success: float | None
    accuracy: float | None
    derived_answers: list[str] | None
    error: str | None = None
    program: str | None = None
    query: str | None = None


_WORKER = r"""
import json
import sys
from pyswip import Prolog

payload = json.loads(sys.stdin.read())
kb_path = payload["kb_path"]
query = payload["query"]

prolog = Prolog()
prolog.consult(kb_path)
rows = list(prolog.query(query))
answers = []
for row in rows:
    for value in row.values():
        answers.append(str(value))
print(json.dumps(answers, ensure_ascii=False))
"""

# Clause terminator: '.' followed by whitespace/EOF, not '..' / '=..' / floats.
_CLAUSE_SPLIT_RE = re.compile(r"\.(?=\s|$)")


def _normalize_field(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_query(query: str) -> str:
    return query.strip().rstrip(".").strip()


def split_program_and_query(thinking: str | None) -> tuple[str | None, str | None]:
    """Split a Cogito ``<think>`` body into (program, query).

    Convention (matches SFT ``format_assistant_content``): the last top-level
    clause is the execution query; everything before it is the KB program.

    Returns ``(None, None)`` when ``thinking`` is empty. Returns
    ``(program, None)`` when fewer than two clauses are present.
    """
    text = _normalize_field(thinking)
    if text is None:
        return None, None

    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(text) if p.strip()]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0] + ".", None

    query = parts[-1] + "."
    program = ".\n".join(parts[:-1]) + "."
    return program, query


def run_query(code: str, query: str, timeout_s: float) -> list[str]:
    """Execute ``query`` against ``code`` in an isolated SWI-Prolog process."""
    query = _normalize_query(query)
    with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False, encoding="utf-8") as handle:
        handle.write(code)
        if not code.rstrip().endswith("."):
            handle.write(".\n")
        kb_path = handle.name

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _WORKER],
            input=json.dumps({"kb_path": kb_path, "query": query}),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Query timed out after {timeout_s}s") from exc
    finally:
        try:
            os.unlink(kb_path)
        except OSError:
            pass

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(detail or f"Prolog worker exited with code {completed.returncode}")

    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError("Prolog worker returned empty stdout")

    # Worker may print SWI warnings on stdout before the JSON line.
    last_line = stdout.splitlines()[-1]
    answers = json.loads(last_line)
    if not isinstance(answers, list):
        raise RuntimeError(f"Unexpected worker payload: {answers!r}")
    return [str(a) for a in answers]


class QueryDerivationAnalyzer:
    """Run query derivation analysis for one (code, query, expected) triple."""

    def __init__(
        self,
        code: str | None,
        query: str | None,
        expected_output: str | None,
        timeout_s: float = 10.0,
    ) -> None:
        self.code = code
        self.query = query
        self.expected_output = expected_output
        self.timeout_s = timeout_s

    def analyze(self) -> QueryDerivationResult:
        return score_query_derivation(
            code=self.code,
            query=self.query,
            expected_output=self.expected_output,
            timeout_s=self.timeout_s,
        )


def score_query_derivation(
    code: str | None,
    query: str | None,
    expected_output: str | None,
    timeout_s: float = 10.0,
) -> QueryDerivationResult:
    """Convenience wrapper used by dataset generation and gold eval."""
    code = _normalize_field(code)
    query = _normalize_field(query)
    expected = _normalize_field(expected_output)

    if code is None or query is None:
        return QueryDerivationResult(
            success=None,
            accuracy=None,
            derived_answers=None,
            error="missing_fields",
            program=code,
            query=query,
        )

    try:
        derived = run_query(code, query, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 — surface any SWI/worker failure as null
        return QueryDerivationResult(
            success=None,
            accuracy=None,
            derived_answers=None,
            error=str(exc),
            program=code,
            query=query,
        )

    if not derived:
        return QueryDerivationResult(
            success=0.0,
            accuracy=None,
            derived_answers=[],
            error="no_solutions",
            program=code,
            query=query,
        )

    success = 1.0
    if expected is None:
        return QueryDerivationResult(
            success=success,
            accuracy=None,
            derived_answers=derived,
            error="missing_fields",
            program=code,
            query=query,
        )

    accuracy = 1.0 if expected in derived else 0.0
    return QueryDerivationResult(
        success=success,
        accuracy=accuracy,
        derived_answers=derived,
        error=None,
        program=code,
        query=query,
    )
