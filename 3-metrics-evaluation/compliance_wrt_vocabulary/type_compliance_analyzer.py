"""
type_compliance_analyzer.py — Evaluate whether the parameter types used in a
Prolog knowledge base comply with a reference ontology, using semantic
embeddings to assess argument-level type conformance.

Background
----------
When an LLM generates a Prolog knowledge base to solve a medical diagnostic
problem, it should use predicates whose argument values match the semantic
roles defined in a reference ontology.  For example, the ontology might
specify that ``symptom/2`` takes a *patient* identifier as its first argument
and a *clinical symptom* as its second argument.  If the generated code
contains ``symptom(john, fever)`` we want to verify that ``john`` is
semantically close to "patient" and ``fever`` is semantically close to
"symptom".

This module performs that verification at three levels of granularity:

1. **Name compliance** — is the predicate name present in the ontology?
2. **Arity compliance** — does the predicate have the expected number of
   arguments?
3. **Type compliance** — for each argument position, is the actual argument
   value semantically close to the expected type label?

Approach
--------
Type compliance is computed via **cosine similarity between sentence
embeddings**.  The expected type label (e.g. ``"symptom"``) and the actual
argument value (e.g. ``"fever"``) are both embedded using a
``SentenceTransformer`` model and their cosine similarity is measured.  A
similarity at or above a configurable threshold means the argument is
considered type-compliant.

The embedding model is loaded **once per analyzer instance** and cached.
All type labels from the ontology are pre-embedded at construction time to
avoid redundant computation.

Parsing
-------
Prolog source code is parsed purely in Python (no SWI-Prolog subprocess).
The parser:

* Strips ``% …`` and ``/* … */`` comments.
* Splits on ``.`` clause terminators (avoiding ``=..`` and floats).
* Identifies the **head** of each clause (the part before ``:-``).
* Extracts the head predicate name and its top-level arguments using a
  nesting-aware argument splitter (handles compound terms like ``f(g(X), Y)``
  correctly).

Only the *head* of each clause is analysed; body predicates are not checked
directly (they will appear as heads in their own facts/rules elsewhere in
the KB and will be checked at that point).

Variable vs. constant treatment
--------------------------------
Both Prolog variables (e.g. ``X``, ``Patient``) and ground atoms (e.g.
``fever``, ``john``) are embedded.  Variable names in well-written Prolog
often carry semantic meaning (``Patient`` signals the patient role better
than an opaque ``X``), so they are kept as-is after normalisation
(underscores replaced with spaces, case preserved).

Not-evaluable cases
-------------------
* Empty or whitespace-only code → ``method = "empty_code"``.
* No clauses found after parsing → ``method = "no_clauses"``.
* Any unexpected exception → ``method = "error"``.

All other cases use ``method = "semantic"``.

Dependencies
------------
* ``sentence-transformers`` (SentenceTransformer)
* ``scikit-learn`` (cosine_similarity via sklearn or numpy)
* Standard library: ``re``, ``json``, ``dataclasses``
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Default paths and constants
# ---------------------------------------------------------------------------

#: Default path to the ontology JSON file (sibling of this module).
DEFAULT_ONTOLOGY_PATH: Path = Path(__file__).parent / "ontology.json"

#: Default SentenceTransformer model.  Fast, high-quality, already a project
#: dependency via ``sentence-transformers``.
DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"

#: Default cosine similarity threshold above which an argument is considered
#: type-compliant.  Tune this using the test notebook.
DEFAULT_THRESHOLD: float = 0.27

#: Sentinel string for the *method* field on success.
METHOD_SEMANTIC = "semantic"

#: Sentinel for empty-code failures.
METHOD_EMPTY_CODE = "empty_code"

#: Sentinel when no clauses could be parsed.
METHOD_NO_CLAUSES = "no_clauses"

#: Sentinel for unexpected runtime errors.
METHOD_ERROR = "error"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ArgumentCompliance:
    """Type-compliance verdict for a single argument of a predicate call.

    Attributes
    ----------
    position:
        0-indexed position of this argument.
    argument_value:
        The raw argument string extracted from the Prolog clause (e.g.
        ``"fever"`` or ``"X"``).
    expected_type_label:
        The semantic type label from the ontology (e.g. ``"symptom"``).
    similarity:
        Cosine similarity between the embedding of *argument_value* and the
        embedding of *expected_type_label*.  In ``[-1.0, 1.0]`` but
        practically in ``[0.0, 1.0]`` for normalised embeddings.
    is_compliant:
        ``True`` if *similarity* ≥ the configured threshold.
    """

    position: int
    argument_value: str
    expected_type_label: str
    similarity: float
    is_compliant: bool

    def __repr__(self) -> str:
        status = "OK" if self.is_compliant else "FAIL"
        return (
            f"ArgumentCompliance(pos={self.position}, "
            f"value={self.argument_value!r}, "
            f"expected={self.expected_type_label!r}, "
            f"sim={self.similarity:.3f}, {status})"
        )


@dataclass
class PredicateCompliance:
    """Type-compliance verdict for a single predicate usage (one clause head).

    Attributes
    ----------
    clause:
        Original clause text (stripped, without trailing ``.``).
    predicate_name:
        The functor of the head predicate (e.g. ``"symptom"``).
    actual_arity:
        Number of top-level arguments in the clause head.
    expected_arity:
        Arity declared in the ontology, or ``None`` if the predicate is not
        in the ontology.
    name_found:
        ``True`` if *predicate_name* exists in the ontology.
    arity_match:
        ``True`` if *actual_arity* == *expected_arity*.  Always ``False``
        when *name_found* is ``False``.
    argument_compliances:
        One :class:`ArgumentCompliance` per argument position.  Empty when
        *name_found* is ``False`` or *arity_match* is ``False`` (argument
        positions are undefined in those cases).
    type_compliance_score:
        Mean cosine similarity across all argument positions, or ``None`` when
        *argument_compliances* is empty.
    """

    clause: str
    predicate_name: str
    actual_arity: int
    expected_arity: Optional[int]
    name_found: bool
    arity_match: bool
    argument_compliances: list[ArgumentCompliance] = field(default_factory=list)

    @property
    def type_compliance_score(self) -> Optional[float]:
        """Mean cosine similarity across all argument positions."""
        if not self.argument_compliances:
            return None
        return float(np.mean([ac.similarity for ac in self.argument_compliances]))

    @property
    def type_compliant_fraction(self) -> Optional[float]:
        """Fraction of arguments whose similarity meets the threshold."""
        if not self.argument_compliances:
            return None
        n_ok = sum(1 for ac in self.argument_compliances if ac.is_compliant)
        return n_ok / len(self.argument_compliances)

    def __repr__(self) -> str:
        if not self.name_found:
            return f"PredicateCompliance({self.predicate_name}/{self.actual_arity}, UNKNOWN)"
        if not self.arity_match:
            return (
                f"PredicateCompliance({self.predicate_name}/{self.actual_arity}, "
                f"ARITY_MISMATCH expected={self.expected_arity})"
            )
        score = self.type_compliance_score
        score_str = f"{score:.3f}" if score is not None else "N/A"
        return (
            f"PredicateCompliance({self.predicate_name}/{self.actual_arity}, "
            f"type_score={score_str})"
        )


@dataclass
class TypeComplianceResult:
    """Full output of :class:`TypeComplianceAnalyzer` for one Prolog program.

    Attributes
    ----------
    predicate_compliances:
        One :class:`PredicateCompliance` for each clause head that was
        successfully parsed.
    unknown_predicates:
        Sorted list of distinct predicate names that appear in the code but
        are not defined in the ontology.
    method:
        How the analysis ended.  One of:

        * ``"semantic"``    — analysis ran successfully.
        * ``"empty_code"``  — input was empty or whitespace-only.
        * ``"no_clauses"``  — no clauses could be parsed.
        * ``"error"``       — unexpected exception.
    error:
        Non-empty description string when *method* is not ``"semantic"``.
    """

    predicate_compliances: list[PredicateCompliance] = field(default_factory=list)
    unknown_predicates: list[str] = field(default_factory=list)
    method: str = METHOD_NO_CLAUSES
    error: str = ""

    # ------------------------------------------------------------------
    # Derived aggregate properties
    # ------------------------------------------------------------------

    @property
    def total_clauses(self) -> int:
        return len(self.predicate_compliances)

    @property
    def known_predicate_clauses(self) -> list[PredicateCompliance]:
        """Clauses whose predicate name was found in the ontology."""
        return [p for p in self.predicate_compliances if p.name_found]

    @property
    def arity_compliant_clauses(self) -> list[PredicateCompliance]:
        """Clauses that are both name-known and arity-correct."""
        return [p for p in self.predicate_compliances if p.arity_match]

    @property
    def name_compliance_score(self) -> float:
        """Fraction of clause heads whose predicate name is in the ontology."""
        if self.total_clauses == 0:
            return 0.0
        return len(self.known_predicate_clauses) / self.total_clauses

    @property
    def arity_compliance_score(self) -> float:
        """Fraction of clause heads with correct predicate name AND arity."""
        if self.total_clauses == 0:
            return 0.0
        return len(self.arity_compliant_clauses) / self.total_clauses

    @property
    def mean_type_compliance_score(self) -> Optional[float]:
        """Mean type compliance score across all arity-correct clause heads.

        Returns ``None`` when no clause passed both name and arity checks
        (making the type score undefined).
        """
        scores = [
            p.type_compliance_score
            for p in self.arity_compliant_clauses
            if p.type_compliance_score is not None
        ]
        if not scores:
            return None
        return float(np.mean(scores))

    def __repr__(self) -> str:
        if self.error:
            return f"TypeComplianceResult(method={self.method!r}, error={self.error!r})"
        mts = self.mean_type_compliance_score
        mts_str = f"{mts:.3f}" if mts is not None else "N/A"
        return (
            f"TypeComplianceResult("
            f"clauses={self.total_clauses}, "
            f"name={self.name_compliance_score:.3f}, "
            f"arity={self.arity_compliance_score:.3f}, "
            f"type={mts_str}, "
            f"unknown={self.unknown_predicates})"
        )


# ---------------------------------------------------------------------------
# Prolog parsing helpers
# ---------------------------------------------------------------------------

# Matches a % line comment until end-of-line.
_LINE_COMMENT_RE = re.compile(r"%[^\n]*")

# Matches a /* ... */ block comment (non-greedy, dotall).
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# Matches a quoted atom '...' or double-quoted string "..." (handles escapes).
_QUOTED_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")

# Clause terminator: a '.' NOT preceded by '.' or '=' and followed by
# whitespace or end-of-string.  Avoids splitting on '=..', floats, or
# qualified calls like 'lists:member'.
_CLAUSE_SEP_RE = re.compile(r"(?<![.=])\.\s*(?=\s|$)")

# Matches a predicate head: functor_name followed by '(' or end-of-token.
# E.g. captures 'symptom' from 'symptom(john, fever)' or
# 'has_condition' from 'has_condition(p1, flu)'.
_HEAD_FUNCTOR_RE = re.compile(
    r"^\s*([a-z_][a-zA-Z0-9_]*)\s*(?:\((.*)\)\s*)?$", re.DOTALL
)


def _strip_comments(code: str) -> str:
    """Remove % line and /* */ block comments from Prolog source."""
    code = _BLOCK_COMMENT_RE.sub(" ", code)
    code = _LINE_COMMENT_RE.sub("", code)
    return code


def _split_clauses(code: str) -> list[str]:
    """Split cleaned Prolog source into individual clause strings.

    The trailing ``.`` is consumed by the split.  Each returned string is
    the clause text without its terminator, stripped of leading/trailing
    whitespace.
    """
    parts = _CLAUSE_SEP_RE.split(code)
    return [p.strip() for p in parts if p.strip()]


def _extract_head(clause: str) -> str:
    """Return the head of a Prolog clause (the part before ``:-``).

    For facts (no ``:-``), the entire clause is the head.
    For rules, everything before the first ``:-`` is the head.
    Handles ``:-`` that may appear inside quoted atoms by a simple
    left-to-right scan respecting parenthesis depth and quotes.
    """
    # Walk character-by-character, tracking depth and quote state,
    # to find the outermost ':-' operator.
    depth = 0
    in_single = False
    in_double = False
    i = 0
    while i < len(clause):
        c = clause[i]
        if in_single:
            if c == "'" and (i == 0 or clause[i - 1] != "\\"):
                in_single = False
        elif in_double:
            if c == '"' and (i == 0 or clause[i - 1] != "\\"):
                in_double = False
        else:
            if c == "'":
                in_single = True
            elif c == '"':
                in_double = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == ":" and depth == 0 and i + 1 < len(clause) and clause[i + 1] == "-":
                # Found top-level ':-'
                return clause[:i].strip()
        i += 1
    return clause.strip()


def _split_top_level_args(args_str: str) -> list[str]:
    """Split a comma-separated argument string respecting nested parentheses
    and quoted atoms/strings.

    For example, ``"john, f(a, b), 'it,is'"`` → ``["john", "f(a, b)", "'it,is'"]``.
    """
    args: list[str] = []
    current: list[str] = []
    depth = 0
    in_single = False
    in_double = False

    for c in args_str:
        if in_single:
            current.append(c)
            if c == "'":
                in_single = False
        elif in_double:
            current.append(c)
            if c == '"':
                in_double = False
        elif c == "'":
            in_single = True
            current.append(c)
        elif c == '"':
            in_double = True
            current.append(c)
        elif c == "(":
            depth += 1
            current.append(c)
        elif c == ")":
            depth -= 1
            current.append(c)
        elif c == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(c)

    # Last argument
    last = "".join(current).strip()
    if last:
        args.append(last)

    return args


def _normalise_value(value: str) -> str:
    """Normalise a Prolog atom/variable for embedding.

    * Strips surrounding whitespace and quotes.
    * Replaces underscores with spaces (``fever_onset`` → ``fever onset``).
    * Leaves case intact — Prolog variables like ``Patient`` carry useful
      semantics and should not be lower-cased.
    """
    value = value.strip().strip("'\"")
    value = value.replace("_", " ")
    return value


def _parse_head_predicate(head: str) -> tuple[str, list[str]] | None:
    """Parse the functor name and top-level arguments from a clause head.

    Returns ``(name, args)`` on success, or ``None`` if the head does not
    look like a standard predicate call (e.g. it is a directive ``:-`` or
    an operator expression).

    Handles:
    * ``symptom(john, fever)``   → ``("symptom", ["john", "fever"])``
    * ``diagnosis(stroke)``      → ``("diagnosis", ["stroke"])``
    * ``loaded``                 → ``("loaded", [])``  (arity-0 atom)
    * ``:-``                     → ``None``  (directives are skipped)
    """
    head = head.strip()

    # Skip directives (:- use_module(...) etc.)
    if head.startswith(":-") or head.startswith("?-"):
        return None

    m = _HEAD_FUNCTOR_RE.match(head)
    if not m:
        return None

    name: str = m.group(1)
    args_str: str | None = m.group(2)

    if args_str is None:
        # Arity-0 fact: just an atom
        return name, []

    args = _split_top_level_args(args_str)
    return name, args


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D numpy vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Main analyser class
# ---------------------------------------------------------------------------


class TypeComplianceAnalyzer:
    """Evaluate parameter-type compliance of a Prolog KB against an ontology.

    The analyser works in two phases:

    1. **Setup** (``__init__``): load the ontology and the embedding model,
       pre-embed all ontology type labels.
    2. **Analysis** (``analyze``): parse the Prolog source, extract clause
       heads, and for each known predicate check argument type compliance via
       cosine similarity.

    Parameters
    ----------
    code:
        Raw Prolog source code (the ``reasoning`` or ``clauses`` column
        value for one program).  Both multi-clause strings and pre-split
        lists of clause strings are accepted.
    ontology:
        The loaded ontology dictionary (as parsed from ``ontology.json``).
        Pass the result of ``load_ontology()`` here.
    model_name:
        Name of the SentenceTransformer model to use for embeddings.
    threshold:
        Cosine similarity threshold above which an argument is considered
        type-compliant (default: ``DEFAULT_THRESHOLD``).
    model:
        An already-loaded ``SentenceTransformer`` instance.  When supplied,
        ``model_name`` is ignored and no new model is loaded.  This is useful
        when analysing many programs in a loop to avoid reloading the model
        every time.
    type_embeddings:
        Pre-computed ontology type embeddings (as built by
        ``_precompute_type_embeddings``).  When supplied, the costly
        ontology-embedding step is skipped.  Pass the ``_type_embeddings``
        attribute of a previously created ``TypeComplianceAnalyzer`` instance.

    Examples
    --------
    >>> ontology = load_ontology()
    >>> analyzer = TypeComplianceAnalyzer(code=prolog_code, ontology=ontology)
    >>> result = analyzer.analyze()
    >>> print(result.mean_type_compliance_score)
    0.712

    To reuse the model across many programs::

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(DEFAULT_MODEL_NAME)
        first = TypeComplianceAnalyzer(code=code1, ontology=ontology, model=model)
        first.analyze()
        # Reuse both model and type embeddings for subsequent calls
        second = TypeComplianceAnalyzer(
            code=code2, ontology=ontology,
            model=model, type_embeddings=first._type_embeddings,
        )
    """

    def __init__(
        self,
        code: str | list[str],
        ontology: dict,
        model_name: str = DEFAULT_MODEL_NAME,
        threshold: float = DEFAULT_THRESHOLD,
        model: "SentenceTransformer | None" = None,
        type_embeddings: "dict[str, dict[int, tuple[str, np.ndarray]]] | None" = None,
    ) -> None:
        # Normalise code to a single string
        if isinstance(code, list):
            self.code: str = "\n".join(code)
        else:
            self.code = code

        self.ontology = ontology
        self.threshold = threshold

        # Re-use a pre-loaded model if supplied, otherwise load a new one
        self._model = model if model is not None else SentenceTransformer(model_name)

        # Re-use pre-computed type embeddings if supplied, otherwise compute them
        # Shape: {predicate_name: {position: (type_label, embedding_vector)}}
        if type_embeddings is not None:
            self._type_embeddings = type_embeddings
        else:
            self._type_embeddings: dict[str, dict[int, tuple[str, np.ndarray]]] = {}
            self._precompute_type_embeddings()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self) -> TypeComplianceResult:
        """Run the full type-compliance analysis.

        Returns
        -------
        TypeComplianceResult
            Always returns a result; never raises.  Check ``result.error``
            for failure details.
        """
        if not self.code or not self.code.strip():
            return TypeComplianceResult(
                method=METHOD_EMPTY_CODE,
                error="Empty or whitespace-only Prolog code.",
            )

        try:
            return self._analyze()
        except Exception as exc:  # noqa: BLE001
            return TypeComplianceResult(
                method=METHOD_ERROR,
                error=f"Unexpected error during analysis: {exc}",
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _precompute_type_embeddings(self) -> None:
        """Embed every type label defined in the ontology once at init time."""
        predicates: dict = self.ontology.get("predicates", {})
        for pred_name, pred_def in predicates.items():
            self._type_embeddings[pred_name] = {}
            for param in pred_def.get("params", []):
                pos: int = param["position"]
                label: str = param["type_label"]
                vec: np.ndarray = self._model.encode(label, convert_to_numpy=True)
                self._type_embeddings[pred_name][pos] = (label, vec)

    def _analyze(self) -> TypeComplianceResult:
        # 1. Strip comments and split into clauses
        clean = _strip_comments(self.code)
        clauses = _split_clauses(clean)

        if not clauses:
            return TypeComplianceResult(
                method=METHOD_NO_CLAUSES,
                error="No Prolog clauses found after parsing.",
            )

        # 2. Analyse each clause head
        predicate_compliances: list[PredicateCompliance] = []
        unknown_names: set[str] = set()
        ontology_predicates: dict = self.ontology.get("predicates", {})

        for clause_text in clauses:
            head_text = _extract_head(clause_text)
            parsed = _parse_head_predicate(head_text)

            if parsed is None:
                # Directive or unrecognised head — skip silently
                continue

            pred_name, args = parsed
            actual_arity = len(args)

            # --- Name check ---
            if pred_name not in ontology_predicates:
                unknown_names.add(pred_name)
                predicate_compliances.append(
                    PredicateCompliance(
                        clause=clause_text,
                        predicate_name=pred_name,
                        actual_arity=actual_arity,
                        expected_arity=None,
                        name_found=False,
                        arity_match=False,
                    )
                )
                continue

            pred_def: dict = ontology_predicates[pred_name]
            expected_arity: int = pred_def["arity"]

            # --- Arity check ---
            if actual_arity != expected_arity:
                predicate_compliances.append(
                    PredicateCompliance(
                        clause=clause_text,
                        predicate_name=pred_name,
                        actual_arity=actual_arity,
                        expected_arity=expected_arity,
                        name_found=True,
                        arity_match=False,
                    )
                )
                continue

            # --- Type compliance check for each argument ---
            arg_compliances: list[ArgumentCompliance] = []
            type_embs = self._type_embeddings.get(pred_name, {})

            for pos, raw_arg in enumerate(args):
                if pos not in type_embs:
                    # No type defined for this position — skip
                    continue

                type_label, type_vec = type_embs[pos]
                normalised = _normalise_value(raw_arg)
                arg_vec: np.ndarray = self._model.encode(
                    normalised, convert_to_numpy=True
                )
                sim = _cosine_similarity(arg_vec, type_vec)

                arg_compliances.append(
                    ArgumentCompliance(
                        position=pos,
                        argument_value=raw_arg,
                        expected_type_label=type_label,
                        similarity=round(sim, 6),
                        is_compliant=sim >= self.threshold,
                    )
                )

            predicate_compliances.append(
                PredicateCompliance(
                    clause=clause_text,
                    predicate_name=pred_name,
                    actual_arity=actual_arity,
                    expected_arity=expected_arity,
                    name_found=True,
                    arity_match=True,
                    argument_compliances=arg_compliances,
                )
            )

        return TypeComplianceResult(
            predicate_compliances=predicate_compliances,
            unknown_predicates=sorted(unknown_names),
            method=METHOD_SEMANTIC,
        )


# ---------------------------------------------------------------------------
# Ontology loader
# ---------------------------------------------------------------------------


def load_ontology(path: str | Path | None = None) -> dict:
    """Load and return the ontology JSON as a Python dictionary.

    Parameters
    ----------
    path:
        Path to the JSON file.  Defaults to ``ontology.json`` in the same
        directory as this module.

    Returns
    -------
    dict
        The parsed ontology dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    if path is None:
        path = DEFAULT_ONTOLOGY_PATH
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
