"""
coherence_analyzer.py — Assess coherence between an LRM's natural-language
answer and the diagnosis derived from its own Prolog knowledge base.

Background
----------
A Large Reasoning Model (LRM) produces two distinct outputs:

* **Reasoning trace** (``reasoning`` column): a Prolog knowledge base (KB)
  that encodes the diagnostic logic.  When queried with ``?- diagnosis(X).``
  it should yield one or more diagnosis atoms (e.g. ``stroke``,
  ``cardiac_arrest``).
* **Natural-language answer** (``answer`` column): the model's stated
  conclusion in plain English.

This module measures whether the two outputs are *coherent* — i.e. whether
the KB's conclusions are reflected in the natural-language answer.

Approach
--------
1. **Prolog query (dynamic, subprocess-based):** the KB is written to a temp
   ``.pl`` file and a fresh ``swipl`` subprocess is spawned.  The query
   ``findall(X, diagnosis(X), Ds), maplist(writeln, Ds), halt.`` collects every
   solution and writes one atom per line to stdout.  Each subprocess gets a
   pristine Prolog engine, avoiding pyswip singleton contamination.

2. **LLM-as-judge (per diagnosis atom):** for each KB-derived diagnosis the
   LLM judge (a LangChain ``BaseChatModel``) is asked:
   "Does the natural-language answer mention or refer to this diagnosis?"
   The judge returns a structured response: ``is_mentioned`` (bool),
   ``confidence`` (0–1 float), and a brief ``explanation``.

3. **Scoring:** the metric is the *proportion* of KB diagnoses that the judge
   considers mentioned in the answer (see ``metrics.py``).

Not-evaluable cases
-------------------
* KB fails to load in SWI-Prolog → ``method = "kb_load_failed"``.
* KB loads but ``diagnosis/1`` yields no solutions → ``method = "query_failed"``.
* Both cases propagate as ``None`` in the metric (see ``metrics.py``).

Dependencies
------------
* SWI-Prolog (``swipl`` on PATH)
* LangChain + a compatible chat model (e.g. ``langchain-openai`` with OpenAI or
  an OpenRouter-compatible endpoint)
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default timeout (seconds) for the SWI-Prolog subprocess.
SWIPL_TIMEOUT: int = 30

#: Sentinel value for the *method* field when the query ran successfully.
METHOD_DYNAMIC = "dynamic"

#: Sentinel value when the KB could not be loaded by SWI-Prolog.
METHOD_KB_LOAD_FAILED = "kb_load_failed"

#: Sentinel value when the KB loaded but ``diagnosis/1`` had no solutions.
METHOD_QUERY_FAILED = "query_failed"

# ---------------------------------------------------------------------------
# Prolog goal executed inside the subprocess.
# We wrap the findall in a catch so we can distinguish load errors from
# query errors.  The goal is passed via -g and we always end with halt.
# ---------------------------------------------------------------------------

_PROLOG_GOAL = (
    # The catch recovery distinguishes two error kinds:
    #   1. existence_error(procedure, diagnosis/_) — predicate simply not defined
    #      → treated as query_failed (the KB is valid, just doesn't define diagnosis/1)
    #   2. any other error — true failure, treated as kb_load_failed
    "( catch("
    "    (findall(X, diagnosis(X), Ds), Ds \\= []),"
    "    Error,"
    "    ( Error = error(existence_error(procedure, diagnosis/_), _)"
    "      -> (write('NO_SOLUTIONS'), nl, halt(2))"
    "      ;  (write('ERROR:'), write(Error), nl, halt(1))"
    "    )"
    "  ) -> "
    "  (write('DIAGNOSES_START'), nl,"
    "   maplist(writeln, Ds),"
    "   write('DIAGNOSES_END'), nl,"
    "   halt(0))"
    "  ; "
    "  (write('NO_SOLUTIONS'), nl, halt(2))"
    ")."
)


# ---------------------------------------------------------------------------
# Pydantic model for structured LLM-judge output
# ---------------------------------------------------------------------------


class JudgeResponse(BaseModel):
    """Structured output from the LLM judge for a single diagnosis."""

    is_mentioned: bool = Field(
        description=(
            "True if the natural-language answer mentions, refers to, or "
            "describes this diagnosis (even using different wording or synonyms)."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your confidence in the is_mentioned judgment, from 0.0 (completely "
            "uncertain) to 1.0 (completely certain)."
        ),
    )
    explanation: str = Field(
        description=(
            "A brief explanation (1-3 sentences) justifying your judgment."
        )
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DiagnosisJudgment:
    """The LLM judge's verdict for one KB-derived diagnosis atom."""

    #: The Prolog atom as returned by the KB (e.g. ``"stroke"``).
    diagnosis: str

    #: Whether the natural-language answer mentions this diagnosis.
    is_mentioned: bool

    #: Judge's confidence in its verdict, in [0.0, 1.0].
    confidence: float

    #: Short free-text explanation from the judge.
    explanation: str


@dataclass
class ConsistencyResult:
    """Full output of :class:`ConsistencyAnalyzer`.

    Fields
    ------
    kb_diagnoses:
        All diagnosis atoms derived from the KB via ``?- diagnosis(X).``
        Empty if the query failed or the KB could not load.
    llm_answer:
        The natural-language answer that was evaluated.
    judgments:
        One :class:`DiagnosisJudgment` per entry in ``kb_diagnoses``.
        Empty when ``kb_diagnoses`` is empty.
    method:
        How the analysis ended:

        * ``"dynamic"``         — KB loaded and query succeeded.
        * ``"kb_load_failed"``  — SWI-Prolog rejected the KB.
        * ``"query_failed"``    — KB loaded but no ``diagnosis/1`` solutions.
    error:
        Non-empty string if something went wrong (SWI-Prolog crash, timeout,
        judge API error, …).  When non-empty, ``judgments`` may be incomplete.
    """

    kb_diagnoses: list[str] = field(default_factory=list)
    llm_answer: str = ""
    judgments: list[DiagnosisJudgment] = field(default_factory=list)
    method: str = METHOD_QUERY_FAILED
    error: str = ""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class ConsistencyAnalyzer:
    """Assess coherence between a Prolog KB and a natural-language answer.

    Parameters
    ----------
    reasoning:
        The Prolog knowledge base produced by the LRM (the ``reasoning``
        column).
    answer:
        The natural-language diagnostic answer produced by the LRM (the
        ``answer`` column).
    llm:
        A LangChain ``BaseChatModel`` used as the judge.  Must support
        ``with_structured_output``.  Configure it externally (e.g.
        ``ChatOpenAI(model="gpt-4o-mini", temperature=0)``).

    Example
    -------
    >>> from langchain_openai import ChatOpenAI
    >>> llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    >>> analyzer = CoherenceAnalyzer(reasoning=kb_code, answer=llm_answer, llm=llm)
    >>> result = analyzer.analyze()
    >>> print(result.kb_diagnoses)
    ['stroke']
    >>> print(result.judgments[0].is_mentioned)
    True
    """

    #: System prompt given to the LLM judge.  Describes its role and the
    #: expected output format (handled by structured output, not by this prompt).
    _JUDGE_SYSTEM_PROMPT = (
        "You are a medical diagnosis coherence judge.  "
        "Your task is to determine whether a natural-language answer "
        "mentions, refers to, or describes a specific medical diagnosis.  "
        "Be generous with paraphrases and synonyms (e.g. 'heart attack' "
        "counts as a mention of 'myocardial_infarction').  "
        "Focus only on whether the diagnosis is present in the answer, "
        "not on whether it is correct."
    )

    #: User prompt template for a single diagnosis evaluation.
    _JUDGE_USER_PROMPT = (
        "Diagnosis to look for: {diagnosis}\n\n"
        "Natural-language answer to evaluate:\n"
        "---\n"
        "{answer}\n"
        "---\n\n"
        "Does the answer mention, refer to, or describe the diagnosis above?"
    )

    def __init__(
        self,
        reasoning: str,
        answer: str,
        llm: BaseChatModel,
        swipl_timeout: int = SWIPL_TIMEOUT,
    ) -> None:
        self.reasoning = reasoning
        self.answer = answer
        self.llm = llm
        self.swipl_timeout = swipl_timeout

        # Build the judge chain once (reused for every diagnosis atom).
        self._judge = llm.with_structured_output(JudgeResponse)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self) -> ConsistencyResult:
        """Run the full coherence analysis.

        Returns
        -------
        ConsistencyResult
            Always returns a result; never raises.  Check ``result.error``
            for failure details.
        """
        result = ConsistencyResult(llm_answer=self.answer)

        # --- Step 1: query the Prolog KB -----------------------------------
        try:
            diagnoses, method, prolog_error = self._query_kb(self.reasoning)
        except Exception as exc:  # noqa: BLE001
            result.method = METHOD_KB_LOAD_FAILED
            result.error = f"Unexpected error during Prolog query: {exc}"
            print(f"Unexpected error during Prolog query: {exc}")
            return result

        diagnoses = list(set(diagnoses))

        result.method = method
        result.kb_diagnoses = diagnoses

        if prolog_error:
            result.error = prolog_error
        if method != METHOD_DYNAMIC:
            # Not evaluable: no diagnoses to judge.
            return result

        # --- Step 2: LLM judge for each diagnosis atom ---------------------
        judge_errors: list[str] = []
        for diag in diagnoses:
            try:
                # print("querying the judge, continue")
                # continue
                judgment = self._judge_diagnosis(diag, self.answer)
            except Exception as exc:  # noqa: BLE001
                # print("judge exception, continue")
                # continue
                judge_errors.append(f"Judge failed for '{diag}': {exc}")
                # Insert a safe fallback judgment so the list stays aligned.
                judgment = DiagnosisJudgment(
                    diagnosis=diag,
                    is_mentioned=False,
                    confidence=0.0,
                    explanation=f"Judge call failed: {exc}",
                )
            result.judgments.append(judgment)

        if judge_errors:
            combined = "; ".join(judge_errors)
            result.error = (result.error + " | " + combined).lstrip(" | ")

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _query_kb(
        self, kb_code: str
    ) -> tuple[list[str], str, str]:
        """Write the KB to a temp file and query it with SWI-Prolog.

        Returns
        -------
        (diagnoses, method, error)
            *diagnoses* — list of diagnosis atoms (may be empty).
            *method*    — one of the METHOD_* constants.
            *error*     — non-empty string on failure, empty on success.
        """
        if not kb_code or not kb_code.strip():
            return [], METHOD_KB_LOAD_FAILED, "Empty reasoning / KB code."

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pl", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(kb_code)
            tmp_path = tmp.name

        try:
            return self._run_swipl(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _run_swipl(
        self, kb_path: str
    ) -> tuple[list[str], str, str]:
        """Spawn swipl, load the KB, run the diagnosis query, parse output."""
        cmd = [
            "swipl",
            "-q",            # suppress banner / informational messages
            "-f", kb_path,   # load the KB file
            "-g", _PROLOG_GOAL,
            "-t", "halt",    # if -g goal fails or is never reached, halt
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.swipl_timeout,
            )
        except subprocess.TimeoutExpired:
            return [], METHOD_KB_LOAD_FAILED, (
                f"SWI-Prolog timed out after {self.swipl_timeout}s."
            )
        except FileNotFoundError:
            return [], METHOD_KB_LOAD_FAILED, (
                "SWI-Prolog ('swipl') not found on PATH."
            )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        # Check for hard load errors in stderr (syntax errors, unknown
        # predicates referenced at load time, etc.)
        if _has_load_error(stderr) and "DIAGNOSES_START" not in stdout:
            return [], METHOD_KB_LOAD_FAILED, (
                f"SWI-Prolog load error: {stderr[:500]}"
            )

        # Explicit error from within our Prolog goal
        if stdout.startswith("ERROR:"):
            return [], METHOD_KB_LOAD_FAILED, (
                f"Prolog goal error: {stdout[:500]}"
            )

        # No solutions
        if "NO_SOLUTIONS" in stdout:
            return [], METHOD_QUERY_FAILED, ""

        # Parse the DIAGNOSES_START … DIAGNOSES_END block
        diagnoses = _parse_diagnoses(stdout)
        if not diagnoses:
            # stdout had DIAGNOSES_START but nothing between them
            return [], METHOD_QUERY_FAILED, (
                "KB loaded but diagnosis/1 returned an empty list."
            )

        return diagnoses, METHOD_DYNAMIC, ""

    def _judge_diagnosis(
        self, diagnosis: str, answer: str
    ) -> DiagnosisJudgment:
        """Ask the LLM judge whether *answer* mentions *diagnosis*."""
        # Normalise the atom for display: replace underscores with spaces so
        # the judge can reason more naturally about the term.
        human_readable = diagnosis.replace("_", " ")

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self._JUDGE_SYSTEM_PROMPT),
            HumanMessage(
                content=self._JUDGE_USER_PROMPT.format(
                    diagnosis=human_readable,
                    answer=answer,
                )
            ),
        ]

        response: JudgeResponse = self._judge.invoke(messages)

        return DiagnosisJudgment(
            diagnosis=diagnosis,
            is_mentioned=response.is_mentioned,
            confidence=response.confidence,
            explanation=response.explanation,
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _has_load_error(stderr: str) -> bool:
    """Return True if stderr contains SWI-Prolog load/syntax error markers."""
    error_markers = ("ERROR:", "Syntax error", "existence_error")
    return any(m in stderr for m in error_markers)


def _parse_diagnoses(stdout: str) -> list[str]:
    """Extract diagnosis atoms from the block between DIAGNOSES_START and
    DIAGNOSES_END in the swipl stdout."""
    match = re.search(
        r"DIAGNOSES_START\s*\n(.*?)\nDIAGNOSES_END",
        stdout,
        re.DOTALL,
    )
    if not match:
        return []

    block = match.group(1).strip()
    if not block:
        return []

    diagnoses = []
    for raw in block.splitlines():
        atom = raw.strip()
        if atom:
            diagnoses.append(atom)
    return diagnoses
