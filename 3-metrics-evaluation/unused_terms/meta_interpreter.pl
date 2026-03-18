% meta_interpreter.pl  (unused-terms experiment, self-contained copy)
% -----------------------------------------------------------------------
% solve(Goal, Proof)  – prove Goal and build a structured proof tree.
% collect_used_goals(+Proof, -Goals) – extract all user-predicate goals
%   that were actually exercised in a proof tree.
% enumerate_kb_clauses(-Heads) – enumerate every user-defined clause head
%   currently loaded in the database (facts AND rule heads).
% -----------------------------------------------------------------------

% ---- solve/2 -----------------------------------------------------------

solve(true, proof(true, builtin, [])).

solve((A, B), proof(conj(A,B), conjunction, [PA,PB])) :-
    solve(A, PA),
    solve(B, PB).

solve((A ; B), proof(disj(A,B), disjunction, [P])) :-
    ( solve(A, P) ; solve(B, P) ).

solve(\+(A), proof(neg(A), negation, [])) :-
    \+ solve(A, _).

solve(!, proof(cut, builtin, [])).

solve(X is Expr, proof(is(X,Expr), builtin, [])) :- X is Expr.

solve(X > Y,   proof(gt(X,Y),       builtin, [])) :- X > Y.
solve(X < Y,   proof(lt(X,Y),       builtin, [])) :- X < Y.
solve(X >= Y,  proof(gte(X,Y),      builtin, [])) :- X >= Y.
solve(X =< Y,  proof(lte(X,Y),      builtin, [])) :- X =< Y.
solve(X =:= Y, proof(arith_eq(X,Y), builtin, [])) :- X =:= Y.
solve(X =\= Y, proof(arith_neq(X,Y),builtin, [])) :- X =\= Y.

solve(X = Y,  proof(unify(X,Y),     builtin, [])) :- X = Y.
solve(X \= Y, proof(not_unify(X,Y), builtin, [])) :- X \= Y.

solve(fail, _) :- fail.

% Regular user-defined predicates
solve(Goal, proof(Goal, rule(Head,Body), SubProofs)) :-
    Goal \= true,
    Goal \= fail,
    Goal \= (_,_),
    Goal \= (_;_),
    Goal \= \+(_),
    Goal \= !,
    Goal \= (_  is _),
    Goal \= (_ >  _),
    Goal \= (_ <  _),
    Goal \= (_ >= _),
    Goal \= (_ =< _),
    Goal \= (_ =:= _),
    Goal \= (_ =\= _),
    Goal \= (_ =  _),
    Goal \= (_ \= _),
    clause(Goal, Body),
    Head = Goal,
    solve(Body, SubProofs).

% ---- collect_used_goals/2 -----------------------------------------------
% Walk a proof tree and accumulate all user-defined goals (i.e. nodes whose
% rule category is rule(...), meaning they came from clause/2).

collect_used_goals(proof(Goal, rule(_,_), SubProof), Goals) :-
    !,
    % Represent this goal as an atom
    term_to_atom(Goal, GoalAtom),
    collect_used_goals(SubProof, SubGoals),
    Goals = [GoalAtom | SubGoals].

collect_used_goals(proof(_, builtin, _), []) :- !.
collect_used_goals(proof(_, negation, _), []) :- !.  % \+ goals: don't recurse

collect_used_goals(proof(_, _, SubProofs), Goals) :-
    % conjunction / disjunction – SubProofs is a list
    is_list(SubProofs),
    !,
    maplist(collect_used_goals, SubProofs, GoalLists),
    append(GoalLists, Goals).

collect_used_goals(proof(_, _, SubProof), Goals) :-
    % SubProof is a single proof term (not a list) – recurse
    collect_used_goals(SubProof, Goals).

collect_used_goals([], []).
collect_used_goals([P|Ps], Goals) :-
    collect_used_goals(P, G1),
    collect_used_goals(Ps, G2),
    append(G1, G2, Goals).

% ---- enumerate_kb_clauses/1 ---------------------------------------------
% Return a list of atoms representing every user-defined clause head.
% We iterate over all predicates in module 'user', skip built-ins and
% imported predicates, and use clause/2 to enumerate their heads.

enumerate_kb_clauses(Heads) :-
    findall(HeadAtom,
        (
            predicate_property(Head, defined),
            \+ predicate_property(Head, built_in),
            \+ predicate_property(Head, imported_from(_)),
            \+ predicate_property(Head, undefined),
            % skip our own meta-interpreter predicates
            functor(Head, Name, _),
            Name \= solve,
            Name \= collect_used_goals,
            Name \= enumerate_kb_clauses,
            clause(Head, _),
            term_to_atom(Head, HeadAtom)
        ),
        Heads).
