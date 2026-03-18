"""
unused_terms_analyzer.py — Detect unused clauses/facts in Prolog programs.

Granularity: **individual clause instances**, not just predicate signatures.
This means that ``symptom(test1)`` is reported as unused even when
``symptom(fever)`` is exercised by the query — something the old
predicate-level approach could not detect.

Approach
--------
1. **Dynamic analysis (primary):** spawn a fresh SWI-Prolog subprocess,
   load the KB, enumerate every user-defined clause head via
   ``predicate_property/2`` + ``clause/2``, run the meta-interpreter on
   ``?- diagnosis(X).``, collect the fully-instantiated goals from the proof
   tree, and compute:
       unused_clauses = all_clauses − used_clauses
   Each subprocess gets a pristine Prolog engine, avoiding pyswip singleton
   contamination.

2. **Static fallback:** if SWI-Prolog cannot load the KB (syntax errors, …),
   a regex-based approximation is used instead (predicate-level, no clause
   enumeration).

3. **Query-failed:** if the KB loads fine but ``diagnosis(X)`` returns no
   solutions, the program is flagged as badly formed.
"""

import re
import sys
import os
import json
import subprocess
import tempfile
import textwrap
from typing import Set
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Path to our local meta-interpreter (self-contained, no dep on predicates_utility)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_META_INTERPRETER_PL = os.path.join(_THIS_DIR, "meta_interpreter.pl")

# ---------------------------------------------------------------------------
# Prolog subprocess script
#
# The entire SWI-Prolog logic is inlined here as a string.  It is executed
# by ``swipl -g main -t halt`` with the KB text passed on stdin.
#
# The script outputs a single JSON line to stdout:
#   {
#     "success": true|false,
#     "all_clauses":    ["head_atom", ...],
#     "used_clauses":   ["head_atom", ...],
#   }
# ---------------------------------------------------------------------------

_PROLOG_SCRIPT = r"""
:- use_module(library(lists)).
:- use_module(library(apply)).

% ---- embedded meta-interpreter (same as meta_interpreter.pl) ---------------

solve(true, proof(true, builtin, [])).
solve((A,B), proof(conj(A,B), conjunction, [PA,PB])) :- solve(A,PA), solve(B,PB).
solve((A;B), proof(disj(A,B), disjunction, [P])) :- (solve(A,P) ; solve(B,P)).
solve(\+(A), proof(neg(A), negation, [])) :- \+ solve(A, _).
solve(!, proof(cut, builtin, [])).
solve(X is E,   proof(is(X,E),       builtin, [])) :- X is E.
solve(X > Y,    proof(gt(X,Y),       builtin, [])) :- X > Y.
solve(X < Y,    proof(lt(X,Y),       builtin, [])) :- X < Y.
solve(X >= Y,   proof(gte(X,Y),      builtin, [])) :- X >= Y.
solve(X =< Y,   proof(lte(X,Y),      builtin, [])) :- X =< Y.
solve(X =:= Y,  proof(arith_eq(X,Y), builtin, [])) :- X =:= Y.
solve(X =\= Y,  proof(arith_neq(X,Y),builtin, [])) :- X =\= Y.
solve(X = Y,    proof(unify(X,Y),    builtin, [])) :- X = Y.
solve(X \= Y,   proof(not_unify(X,Y),builtin, [])) :- X \= Y.
solve(fail, _) :- fail.
solve(Goal, proof(Goal, rule(Head,Body), SubProofs)) :-
    Goal \= true, Goal \= fail,
    Goal \= (_,_), Goal \= (_;_), Goal \= \+(_), Goal \= !,
    Goal \= (_ is _), Goal \= (_ > _), Goal \= (_ < _),
    Goal \= (_ >= _), Goal \= (_ =< _), Goal \= (_ =:= _), Goal \= (_ =\= _),
    Goal \= (_ = _), Goal \= (_ \= _),
    clause(Goal, Body),
    Head = Goal,
    solve(Body, SubProofs).

% ---- collect used goals from proof tree ------------------------------------

collect_used_goals(proof(Goal, rule(_,_), SubProof), Goals) :-
    !,
    term_to_atom(Goal, GoalAtom),
    collect_used_goals(SubProof, SubGoals),
    Goals = [GoalAtom | SubGoals].
collect_used_goals(proof(_, builtin, _), []) :- !.
collect_used_goals(proof(_, negation, _), []) :- !.
collect_used_goals(proof(_, _, SubProofs), Goals) :-
    is_list(SubProofs), !,
    maplist(collect_used_goals, SubProofs, GoalLists),
    append(GoalLists, Goals).
collect_used_goals(proof(_, _, SubProof), Goals) :-
    collect_used_goals(SubProof, Goals).
collect_used_goals([], []).
collect_used_goals([P|Ps], Goals) :-
    collect_used_goals(P, G1),
    collect_used_goals(Ps, G2),
    append(G1, G2, Goals).

% ---- main ------------------------------------------------------------------

:- meta_interpreter_loaded.   % just a marker – defined below
meta_interpreter_loaded.

main :-
    % 1. Read KB from stdin
    read_term_from_atom('', _, []), % warm up
    read_string(user_input, _, KBString),

    % 2. Write KB to a temp file and consult it
    tmp_file_stream(text, TmpFile, Stream),
    write(Stream, KBString),
    close(Stream),

    % Attempt to load; catch syntax / load errors
    ( catch(consult(TmpFile), _, fail) ->
        KBLoaded = true
    ;
        KBLoaded = false
    ),

    ( KBLoaded = false ->
        % Can't load: output failure signal so Python falls back to static
        Result = json([success=false, load_error=true,
                       all_clauses=[], used_clauses=[]]),
        with_output_to(atom(JSON), json_write(current_output, Result, [])),
        writeln(JSON), !
    ;
        true
    ),

    % 3. Enumerate all user-defined clause heads from the KB file
    findall(HeadAtom,
        (
            predicate_property(Head, file(TmpFile)),
            \+ predicate_property(Head, built_in),
            \+ predicate_property(Head, imported_from(_)),
            functor(Head, Name, _),
            % exclude our meta-interpreter predicates
            Name \= solve,
            Name \= collect_used_goals,
            Name \= meta_interpreter_loaded,
            clause(Head, _),
            term_to_atom(Head, HeadAtom)
        ),
        AllClauses),

    % 4. Try to run the query through the meta-interpreter
    ( findall(Goals,
          ( solve(diagnosis(_), Proof),
            collect_used_goals(Proof, Goals) ),
          GoalLists),
      GoalLists \= []
    ->
        QuerySucceeded = true,
        append(GoalLists, Dup),
        sort(Dup, UsedClauses)
    ;
        QuerySucceeded = false,
        UsedClauses = []
    ),

    % 5. Output JSON
    term_to_atom(AllClauses, AllAtom),
    term_to_atom(UsedClauses, UsedAtom),
    format('{"success":~w,"all_clauses":~w,"used_clauses":~w}~n',
           [QuerySucceeded, AllAtom, UsedAtom]).
"""

# ---------------------------------------------------------------------------
# The actual subprocess entry: we write a small wrapper script that loads
# our Prolog logic and calls main/0.  We avoid relying on the external
# meta_interpreter.pl file (everything is inlined above in _PROLOG_SCRIPT).
# The JSON output is simpler to parse than Prolog terms, so we produce it
# directly via format/2 rather than using library(http/json).
# ---------------------------------------------------------------------------

_SWIPL_WRAPPER = textwrap.dedent("""\
    :- use_module(library(lists)).
    :- use_module(library(apply)).

    solve(true, proof(true, builtin, [])).
    solve((A,B), proof(conj(A,B), conjunction, [PA,PB])) :- solve(A,PA), solve(B,PB).
    solve((A;B), proof(disj(A,B), disjunction, [P])) :- (solve(A,P) ; solve(B,P)).
    solve(\\+(A), proof(neg(A), negation, [])) :- \\+ solve(A, _).
    solve(!, proof(cut, builtin, [])).
    solve(X is E,   proof(is(X,E),       builtin, [])) :- X is E.
    solve(X > Y,    proof(gt(X,Y),       builtin, [])) :- X > Y.
    solve(X < Y,    proof(lt(X,Y),       builtin, [])) :- X < Y.
    solve(X >= Y,   proof(gte(X,Y),      builtin, [])) :- X >= Y.
    solve(X =< Y,   proof(lte(X,Y),      builtin, [])) :- X =< Y.
    solve(X =:= Y,  proof(arith_eq(X,Y), builtin, [])) :- X =:= Y.
    solve(X =\\= Y,  proof(arith_neq(X,Y),builtin, [])) :- X =\\= Y.
    solve(X = Y,    proof(unify(X,Y),    builtin, [])) :- X = Y.
    solve(X \\= Y,   proof(not_unify(X,Y),builtin, [])) :- X \\= Y.
    solve(fail, _) :- fail.
    solve(Goal, proof(Goal, rule(Head,Body), SubProofs)) :-
        Goal \\= true, Goal \\= fail,
        Goal \\= (_,_), Goal \\= (_;_), Goal \\= \\+(_), Goal \\= !,
        Goal \\= (_ is _), Goal \\= (_ > _), Goal \\= (_ < _),
        Goal \\= (_ >= _), Goal \\= (_ =< _), Goal \\= (_ =:= _), Goal \\= (_ =\\= _),
        Goal \\= (_ = _), Goal \\= (_ \\= _),
        clause(Goal, Body),
        Head = Goal,
        solve(Body, SubProofs).

    collect_used_goals(proof(Goal, rule(_,_), SubProof), Goals) :-
        !, term_to_atom(Goal, GoalAtom),
        collect_used_goals(SubProof, SubGoals),
        Goals = [GoalAtom | SubGoals].
    collect_used_goals(proof(_, builtin, _), []) :- !.
    collect_used_goals(proof(_, negation, _), []) :- !.
    collect_used_goals(proof(_, _, SubProofs), Goals) :-
        is_list(SubProofs), !,
        maplist(collect_used_goals, SubProofs, GoalLists),
        append(GoalLists, Goals).
    collect_used_goals(proof(_, _, SubProof), Goals) :-
        collect_used_goals(SubProof, Goals).
    collect_used_goals([], []).
    collect_used_goals([P|Ps], Goals) :-
        collect_used_goals(P, G1),
        collect_used_goals(Ps, G2),
        append(G1, G2, Goals).

    main :-
        read_string(user_input, _, KBString),
        tmp_file_stream(text, TmpFile, Stream),
        write(Stream, KBString),
        close(Stream),
        ( catch(consult(TmpFile), _, fail) ->
            true
        ;
            format('{"success":false,"load_error":true,"all_clauses":[],"used_clauses":[]}~n'),
            halt
        ),
        findall(HeadAtom,
            ( predicate_property(Head, file(TmpFile)),
              \\+ predicate_property(Head, built_in),
              \\+ predicate_property(Head, imported_from(_)),
              functor(Head, Name, _),
              Name \\= solve,
              Name \\= collect_used_goals,
              clause(Head, _),
              term_to_atom(Head, HeadAtom) ),
            AllClauses),
        ( catch(
              ( findall(Goals,
                    ( solve(diagnosis(_), Proof),
                      collect_used_goals(Proof, Goals) ),
                    GoalLists),
                GoalLists \\= [] ),
              _, fail)
        ->  QuerySucceeded = true,
            append(GoalLists, Dup), sort(Dup, UsedClauses)
        ;   QuerySucceeded = false, UsedClauses = []
        ),
        with_output_to(atom(AllJSON),  json_write_list(AllClauses)),
        with_output_to(atom(UsedJSON), json_write_list(UsedClauses)),
        format('{"success":~w,"load_error":false,"all_clauses":~w,"used_clauses":~w}~n',
               [QuerySucceeded, AllJSON, UsedJSON]).

    json_write_list(List) :-
        write('['),
        json_write_items(List),
        write(']').
    json_write_items([]).
    json_write_items([H]) :- !,
        write('"'),
        atom_string(H, S),
        % escape backslashes and double-quotes
        atomic_list_concat(Parts, '\\\\', S), atomic_list_concat(Parts, '\\\\\\\\', S2),
        atomic_list_concat(Parts2, '"', S2), atomic_list_concat(Parts2, '\\\\"', S3),
        write(S3),
        write('"').
    json_write_items([H|T]) :-
        json_write_items([H]),
        write(','),
        json_write_items(T).

    :- main.
""")


# ---------------------------------------------------------------------------
# Python-level JSON escaping is simpler — let's just use Python to produce
# the JSON by asking Prolog to write each atom on its own line and parsing
# from there.  We use a cleaner output format.
# ---------------------------------------------------------------------------

_SWIPL_SCRIPT = textwrap.dedent(r"""
    :- use_module(library(lists)).
    :- use_module(library(apply)).

    solve(true, proof(true, builtin, [])).
    solve((A,B), proof(conj(A,B), conjunction, [PA,PB])) :- solve(A,PA), solve(B,PB).
    solve((A;B), proof(disj(A,B), disjunction, [P])) :- (solve(A,P) ; solve(B,P)).
    solve(\+(A), proof(neg(A), negation, [])) :- \+ solve(A, _).
    solve(!, proof(cut, builtin, [])).
    solve(X is E,   proof(is(X,E),       builtin, [])) :- X is E.
    solve(X > Y,    proof(gt(X,Y),       builtin, [])) :- X > Y.
    solve(X < Y,    proof(lt(X,Y),       builtin, [])) :- X < Y.
    solve(X >= Y,   proof(gte(X,Y),      builtin, [])) :- X >= Y.
    solve(X =< Y,   proof(lte(X,Y),      builtin, [])) :- X =< Y.
    solve(X =:= Y,  proof(arith_eq(X,Y), builtin, [])) :- X =:= Y.
    solve(X =\= Y,  proof(arith_neq(X,Y),builtin, [])) :- X =\= Y.
    solve(X = Y,    proof(unify(X,Y),    builtin, [])) :- X = Y.
    solve(X \= Y,   proof(not_unify(X,Y),builtin, [])) :- X \= Y.
    solve(fail, _) :- fail.
    solve(Goal, proof(Goal, rule(Head,Body), SubProofs)) :-
        Goal \= true, Goal \= fail,
        Goal \= (_,_), Goal \= (_;_), Goal \= \+(_), Goal \= !,
        Goal \= (_ is _), Goal \= (_ > _), Goal \= (_ < _),
        Goal \= (_ >= _), Goal \= (_ =< _), Goal \= (_ =:= _), Goal \= (_ =\= _),
        Goal \= (_ = _), Goal \= (_ \= _),
        clause(Goal, Body),
        Head = Goal,
        solve(Body, SubProofs).

    collect_used_goals(proof(Goal, rule(_,_), SubProof), Goals) :-
        !, normalize_term_atom(Goal, GoalAtom),
        collect_used_goals(SubProof, SubGoals),
        Goals = [GoalAtom | SubGoals].
    collect_used_goals(proof(_, builtin, _), []) :- !.
    collect_used_goals(proof(_, negation, _), []) :- !.
    collect_used_goals(proof(_, _, SubProofs), Goals) :-
        is_list(SubProofs), !,
        maplist(collect_used_goals, SubProofs, GoalLists),
        append(GoalLists, Goals).
    collect_used_goals(proof(_, _, SubProof), Goals) :-
        collect_used_goals(SubProof, Goals).
    collect_used_goals([], []).
    collect_used_goals([P|Ps], Goals) :-
        collect_used_goals(P, G1),
        collect_used_goals(Ps, G2),
        append(G1, G2, Goals).

    % Normalize a term by replacing unbound variables with '_'
    % (uses numbervars to give variables canonical names like A, B, ...)
    normalize_term_atom(Term, Atom) :-
        copy_term(Term, Copy),
        numbervars(Copy, 0, _),
        term_to_atom(Copy, Atom).

    % Output helpers: write one atom-per-line between markers
    write_list_section(Tag, List) :-
        format('BEGIN_~w~n', [Tag]),
        forall(member(Item, List),
               ( atom_string(Item, S), writeln(S) )),
        format('END_~w~n', [Tag]).

    main :-
        read_string(user_input, _, KBString),
        tmp_file_stream(text, TmpFile, Stream),
        write(Stream, KBString),
        close(Stream),
        ( catch(consult(TmpFile), _, fail) ->
            true
        ;
            writeln('STATUS:load_error'), halt
        ),
        findall(HeadAtom,
            ( predicate_property(Head, file(TmpFile)),
              \+ predicate_property(Head, built_in),
              \+ predicate_property(Head, imported_from(_)),
              functor(Head, Name, _),
              Name \= solve,
              Name \= collect_used_goals,
              clause(Head, _),
              normalize_term_atom(Head, HeadAtom) ),
            AllClauses),
        ( catch(
              ( findall(Goals,
                    ( solve(diagnosis(_), Proof),
                      collect_used_goals(Proof, Goals) ),
                    GoalLists),
                GoalLists \= [] ),
              _, fail)
        ->  append(GoalLists, Dup), sort(Dup, UsedClauses),
            writeln('STATUS:success')
        ;   UsedClauses = [],
            writeln('STATUS:query_failed')
        ),
        write_list_section('ALL', AllClauses),
        write_list_section('USED', UsedClauses).

    :- main.
""")


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _run_swipl(kb_text: str, timeout: int = 60) -> dict:
    """
    Spawn a fresh ``swipl`` process, feed it ``kb_text`` on stdin, and parse
    its structured output.

    Returns a dict with keys:
      - ``status``: ``"success"`` | ``"query_failed"`` | ``"load_error"``
      - ``all_clauses``: list of strings
      - ``used_clauses``: list of strings
    """
    # Write the Prolog script to a temp file (avoids shell quoting issues)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pl", delete=False, encoding="utf-8"
    ) as f:
        f.write(_SWIPL_SCRIPT)
        script_path = f.name

    try:
        proc = subprocess.run(
            ["swipl", "-g", "true", script_path],
            input=kb_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        os.unlink(script_path)

    stdout = proc.stdout
    if not stdout.strip():
        raise RuntimeError(
            f"swipl produced no output (exit {proc.returncode}).\n"
            f"stderr: {proc.stderr[:500]}"
        )

    return _parse_swipl_output(stdout)


def _parse_swipl_output(text: str) -> dict:
    """Parse the structured text output from the Prolog main/0 predicate."""
    lines = text.splitlines()

    status = "unknown"
    all_clauses: list = []
    used_clauses: list = []

    current_section = None
    current_items: list = []

    for line in lines:
        line = line.rstrip()
        if line.startswith("STATUS:"):
            status = line[len("STATUS:"):]
        elif line.startswith("BEGIN_"):
            current_section = line[len("BEGIN_"):]
            current_items = []
        elif line.startswith("END_"):
            if current_section == "ALL":
                all_clauses = current_items[:]
            elif current_section == "USED":
                used_clauses = current_items[:]
            current_section = None
        elif current_section is not None:
            if line:
                current_items.append(line)

    return {
        "status": status,
        "all_clauses": all_clauses,
        "used_clauses": used_clauses,
    }


# ---------------------------------------------------------------------------
# Static (regex) fallback — kept for KB-load failures
# ---------------------------------------------------------------------------

def _static_extract_predicates(code: str) -> Set[str]:
    predicates: Set[str] = set()
    clean = re.sub(r"%[^\n]*", "", code)
    pattern = re.compile(r"\b([a-z_][a-zA-Z0-9_]*)\s*\(")
    for m in pattern.finditer(clean):
        pred_name = m.group(1)
        rest = clean[m.end() - 1:]
        try:
            arity = _count_args(_extract_args(rest))
            predicates.add(f"{pred_name}/{arity}")
        except Exception:
            continue
    return predicates


def _extract_args(text: str) -> str:
    if not text.startswith("("):
        return ""
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[1:i]
    return ""


def _count_args(args_text: str) -> int:
    if not args_text.strip():
        return 0
    depth = 0
    count = 1
    for ch in args_text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            count += 1
    return count


def _static_extract_called_predicates(code: str) -> Set[str]:
    called: Set[str] = set()
    clean = re.sub(r"%[^\n]*", "", code)
    bodies = []
    for clause in re.split(r"\.\s*(?=\n|$)", clean):
        if ":-" in clause:
            _, body = clause.split(":-", 1)
            bodies.append(body)
    pattern = re.compile(r"\b([a-z_][a-zA-Z0-9_]*)\s*\(")
    for body in bodies:
        for m in pattern.finditer(body):
            pred_name = m.group(1)
            rest = body[m.end() - 1:]
            try:
                arity = _count_args(_extract_args(rest))
                called.add(f"{pred_name}/{arity}")
            except Exception:
                continue
    return called


_PROLOG_BUILTINS = frozenset({
    "is/2", "true/0", "fail/0", "false/0", "halt/0", "halt/1",
    "assert/1", "assertz/1", "asserta/1", "retract/1", "abolish/1",
    "functor/3", "arg/3", "copy_term/2", "nl/0", "write/1", "writeln/1",
    "read/1", "not/1", "call/1", "call/2", "call/3",
    "findall/3", "bagof/3", "setof/3", "forall/2", "aggregate_all/3",
    "msort/2", "sort/2", "length/2", "append/3", "member/2", "memberchk/2",
    "last/2", "nth0/3", "nth1/3", "flatten/2",
    "number_codes/2", "number_chars/2", "atom_codes/2", "atom_chars/2",
    "atom_length/2", "atom_concat/3", "atom_string/2",
    "string_concat/3", "string_length/2", "string_lower/2", "string_upper/2",
    "number_string/2", "term_to_atom/2",
    "format/1", "format/2", "succ/2", "plus/3", "between/3",
    "max_list/2", "min_list/2", "sum_list/2", "numlist/3",
    "pairs_keys/2", "pairs_values/2", "pairs_keys_values/3",
    "maplist/2", "maplist/3", "maplist/4",
    "include/3", "exclude/3", "foldl/4", "foldl/5", "aggregate/3",
})


def _clean_varnames(atom: str) -> str:
    """Replace Prolog numbervars output like '$VAR'(0) with A, '$VAR'(1) with B, etc."""
    def _replace(m: re.Match) -> str:
        n = int(m.group(1))
        # Convert to variable name: A, B, ..., Z, A1, B1, ...
        letter = chr(ord('A') + (n % 26))
        suffix = str(n // 26) if n >= 26 else ""
        return f"{letter}{suffix}"
    return re.sub(r"'\$VAR'\((\d+)\)", _replace, atom)


def _sig(head_atom: str) -> str:
    """Convert a clause head atom string to name/arity signature."""
    m = re.match(r"([a-z_][a-zA-Z0-9_]*)\s*\(", head_atom)
    if m:
        rest = head_atom[m.end() - 1:]
        try:
            arity = _count_args(_extract_args(rest))
            return f"{m.group(1)}/{arity}"
        except Exception:
            pass
    # atom with no args (arity 0)
    m0 = re.match(r"([a-z_][a-zA-Z0-9_]*)$", head_atom.strip())
    if m0:
        return f"{m0.group(1)}/0"
    return head_atom


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class UnusedTermsResult:
    """Result of unused-terms analysis for a single Prolog program."""

    # --- Clause-level (primary, new) ----------------------------------------
    all_clauses: Set[str] = field(default_factory=set)
    """All individual clause/fact heads found in the KB (as strings)."""

    used_clauses: Set[str] = field(default_factory=set)
    """Clause heads exercised during query resolution."""

    unused_clauses: Set[str] = field(default_factory=set)
    """Clause heads defined but never exercised."""

    # --- Predicate-level (derived / backward-compat) ------------------------
    all_predicates: Set[str] = field(default_factory=set)
    """All predicate signatures (name/arity) in the KB."""

    used_predicates: Set[str] = field(default_factory=set)
    """Predicate signatures exercised during query resolution."""

    unused_predicates: Set[str] = field(default_factory=set)
    """Predicate signatures never exercised."""

    # --- Summary fields -------------------------------------------------------
    has_unused_terms: bool = False
    """True if at least one clause is unused."""

    unused_ratio: float = 0.0
    """Fraction of *clauses* that are unused (0.0–1.0)."""

    query: str = ""
    method: str = "dynamic"
    """
    'dynamic'      – SWI-Prolog traced the query successfully.
    'static'       – KB could not be loaded; regex fallback used.
    'query_failed' – KB loaded but query returned no solutions.
    'none'         – Empty code or both methods failed.
    """
    error: str = ""

    def __repr__(self) -> str:
        if self.error and self.method == "none":
            return (
                f"UnusedTermsResult(has_unused={self.has_unused_terms}, "
                f"method={self.method!r}, error={self.error!r})"
            )
        if self.method == "query_failed":
            return (
                f"UnusedTermsResult(has_unused={self.has_unused_terms}, "
                f"method={self.method!r}, error={self.error!r})"
            )
        return (
            f"UnusedTermsResult("
            f"has_unused={self.has_unused_terms}, "
            f"unused_clauses={sorted(self.unused_clauses)}, "
            f"ratio={self.unused_ratio:.2f}, "
            f"method={self.method!r})"
        )


# ---------------------------------------------------------------------------
# Static fallback analyser (predicate-level, used only when KB won't load)
# ---------------------------------------------------------------------------

def _static_analyze(code: str) -> UnusedTermsResult:
    all_preds = _static_extract_predicates(code) - _PROLOG_BUILTINS
    called_preds = _static_extract_called_predicates(code)
    diagnosis_preds = {p for p in all_preds if p.startswith("diagnosis/")}
    used = called_preds | diagnosis_preds
    unused = all_preds - used
    ratio = len(unused) / len(all_preds) if all_preds else 0.0
    return UnusedTermsResult(
        all_predicates=all_preds,
        used_predicates=all_preds & used,
        unused_predicates=unused,
        has_unused_terms=len(unused) > 0,
        unused_ratio=ratio,
        query="(static – no query executed)",
        method="static",
    )


# ---------------------------------------------------------------------------
# Main analyser class
# ---------------------------------------------------------------------------

class UnusedTermsAnalyzer:
    """
    Detect unused clauses/facts in a Prolog program at clause-level granularity.

    Usage::

        result = UnusedTermsAnalyzer(code).analyze()

    The analyzer spawns a fresh ``swipl`` subprocess for each KB (avoiding
    pyswip singleton contamination), loads the KB, enumerates every user-defined
    clause head, runs the standard query ``diagnosis(X)`` through the built-in
    meta-interpreter, collects all goals that appeared in the proof, and
    computes the diff.

    Parameters
    ----------
    code:
        Prolog source code as a string.
    query:
        Prolog query to trace.  Defaults to ``"diagnosis(X)"``.
    python_exe:
        Ignored (kept for backward compatibility with old call sites that
        passed ``python_exe=...``).  The new implementation calls ``swipl``
        directly, not Python.
    """

    DEFAULT_QUERY = "diagnosis(X)"

    def __init__(self, code: str, query: str = DEFAULT_QUERY, python_exe: str = ""):
        self.code = code.strip()
        self.query = query  # stored for reference; currently always diagnosis(_)

    def analyze(self) -> UnusedTermsResult:
        """Run the analysis and return an :class:`UnusedTermsResult`."""
        if not self.code:
            return UnusedTermsResult(
                has_unused_terms=False,
                error="Empty code",
                method="none",
            )

        # --- dynamic path ---
        try:
            raw = _run_swipl(self.code)
        except Exception as exc:
            # swipl not found or subprocess crashed — try static fallback
            try:
                return _static_analyze(self.code)
            except Exception as exc2:
                return UnusedTermsResult(
                    has_unused_terms=False,
                    error=f"Dynamic failed: {exc}; static failed: {exc2}",
                    method="none",
                )

        status = raw["status"]

        if status == "load_error":
            # KB could not be loaded — static fallback
            try:
                return _static_analyze(self.code)
            except Exception as exc:
                return UnusedTermsResult(
                    has_unused_terms=False,
                    error=f"Static fallback failed: {exc}",
                    method="none",
                )

        all_clauses = {_clean_varnames(c) for c in raw["all_clauses"]}
        used_clauses = {_clean_varnames(c) for c in raw["used_clauses"]}

        if status == "query_failed":
            return UnusedTermsResult(
                has_unused_terms=False,
                all_clauses=all_clauses,
                used_clauses=set(),
                unused_clauses=set(),
                all_predicates={_sig(c) for c in all_clauses} - _PROLOG_BUILTINS,
                used_predicates=set(),
                unused_predicates=set(),
                unused_ratio=0.0,
                query=self.query,
                method="query_failed",
                error=f"Query '{self.query}' returned no solutions",
            )

        # success
        # Match used clauses against all_clauses:
        # - Ground facts (no free variables): exact string match.
        # - Rule heads with variables (e.g. diagnosis('$VAR'(0))): match by
        #   functor/arity signature, since the proof records instantiated atoms
        #   like diagnosis(flu) which won't equal the enumerated head atom.
        used_sigs = {_sig(c) for c in used_clauses}

        def _clause_has_vars(atom: str) -> bool:
            """True if the atom string contains a Prolog variable placeholder."""
            return bool(re.search(r"_G\d+|_\d+|\'\$VAR\'\(\d+\)|[A-Z][A-Za-z0-9_]*", atom))

        effective_used_clauses = {
            c for c in all_clauses
            if c in used_clauses
            or (_clause_has_vars(c) and _sig(c) in used_sigs)
        }
        unused_clauses = all_clauses - effective_used_clauses
        ratio = len(unused_clauses) / len(all_clauses) if all_clauses else 0.0

        all_preds = {_sig(c) for c in all_clauses} - _PROLOG_BUILTINS
        used_preds = {_sig(c) for c in effective_used_clauses} - _PROLOG_BUILTINS
        unused_preds = all_preds - used_preds

        return UnusedTermsResult(
            all_clauses=all_clauses,
            used_clauses=effective_used_clauses,
            unused_clauses=unused_clauses,
            all_predicates=all_preds,
            used_predicates=used_preds,
            unused_predicates=unused_preds,
            has_unused_terms=len(unused_clauses) > 0,
            unused_ratio=ratio,
            query=self.query,
            method="dynamic",
        )
