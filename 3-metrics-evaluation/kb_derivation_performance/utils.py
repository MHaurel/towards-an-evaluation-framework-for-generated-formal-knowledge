import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from swiplParser.swiplParser import evaluate_prolog_syntax_per_clause
from tqdm import tqdm

## DATA PROCESSING

def load_results(path, only_perfect=True, enforce=False):
    df = pd.read_csv(path)
    save_new_file = False

    if enforce or not 'clause_per_clause_reward' in df.columns:
        tqdm.pandas()
        df['clause_per_clause_reward'] = df['reasoning'].progress_apply(evaluate_prolog_syntax_per_clause)
        save_new_file = True

    if only_perfect:
        df = df[df['clause_per_clause_reward'] == 1]

    df['clauses'] = df['reasoning'].apply(split_prolog_clauses)
    all_clauses = [c for lst in df['clauses'] for c in lst]
    print(f"There are {len(all_clauses)} clauses in total.")

    if save_new_file:
        df.to_csv(f"{path.replace('.csv', '')}-computed.csv", index=False) # saving the dataframe

    return df

def get_facts_rules_dfs(df):
    facts = [
        {
            "question_index": row['question_index'],
            "model": row['model'],
            "fact": c
        }
        for _, row in df.iterrows()
        for c in row['clauses'] if not is_rule(c)
    ]

    rules = [
        {
            "question_index": row['question_index'],
            "model": row['model'],
            "rule": c
        }
        for _, row in df.iterrows()
        for c in row['clauses'] if is_rule(c)
    ]

    df_facts = pd.DataFrame(facts)
    df_rules = pd.DataFrame(rules)
    return df_facts, df_rules

def get_rules_info(df):
    df_facts, df_rules = get_facts_rules_dfs(df)

    predicate_counts = Counter()
    predicate_positions = []

    for fact in df_facts['fact']:
        s = signature(fact)
        predicate_counts[s] += 1
        predicate_positions.append((s, 'fact'))

    rules_info = []

    for _, row in df_rules.iterrows():
        rule = row['rule']
        model = row['model']
        qi = row['question_index']
        
        try:
            head, body = parse_rule(rule)
            head_sig = signature(head)
            predicate_counts[head_sig] += 1
            predicate_positions.append((head_sig, "rule_head"))

            body_sigs = []
            for b in body:
                sig = signature(b)
                predicate_counts[sig] += 1
                predicate_positions.append((sig, "rule_body"))
                body_sigs.append(sig)

            rules_info.append({
                "question_index": qi,
                "model": model,
                "rule": rule,
                "head": head_sig,
                "body": body_sigs,
                "body_len": len(body_sigs)
            })
        except ValueError as e:
            pass
    
    rules_info_df = pd.DataFrame(rules_info)
    return rules_info_df, rules_info

## PLOTTING

def plot_dependency_graph(df, n_top_nodes=10, save_filename=None):
    G = nx.DiGraph()

    rules_info_df, rules_info = get_rules_info(df)

    for r in rules_info:
        head = r['head']
        for b in r['body']:
            G.add_edge(b, head)

    print(f"Nodes: {len(G.nodes())}")
    print(f"Edges: {len(G.edges())}")

    degree = dict(G.degree())
    top_nodes = sorted(degree, key=degree.get, reverse=True)[:n_top_nodes]

    subG = G.subgraph(top_nodes)
    plt.figure(figsize=(50, 40))
    pos = nx.spring_layout(subG, seed=42)
    nx.draw(subG, pos, with_labels=True, node_size=1500)

    if save_filename is not None:
        plt.savefig(save_filename)
        print(f"Saved to {save_filename}")

    plt.show()







## UTILITARY FUNCTIONS

### CLAUSES

def split_prolog_clauses(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r'%.*', '', text)  # remove comments
    clauses = [c.strip() for c in text.split('.') if c.strip()]
    return clauses

def is_rule(clause):
    return ":-" in clause

def extract_predicate_name(premise):
    return premise.split("(")[0].strip()

def extract_all_predicates_name(line):
    pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?=\()")
    return [m.group(1) for m in pattern.finditer(line)]

# def analyze_body(parts):
#     atomic = len(parts)
#     disjunctions = sum(1 for p in parts if ";" in p)
#     return atomic, disjunctions

### PREDICATE PARSING

def extract_predicate_and_args(term):
    term = term.strip()
    m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)$', term, re.DOTALL)
    if m:
        name = m.group(1)
        args = [a.strip() for a in m.group(2).split(',')]
        return name, args
    return term, []

def signature(term):
    name, args = extract_predicate_and_args(term)
    return f"{name}/{len(args)}"

def parse_rule(rule):
    head, body = rule.split(":-")
    head = head.strip()
    body_parts = [b.strip() for b in body.split(",")]

    # Expand disjunctions A ; B
    expanded = []
    for part in body_parts:
        if ";" in part:
            for dis in part.split(";"):
                expanded.append(dis.strip())
        else:
            expanded.append(part)
    return head, expanded

def classify_rule_body_premises(rule_str):
    """
    Parses a Prolog rule string and classifies each premise.
    TODO: add the conjuctions and the disjunction maybe.
    """
    # 1. Extract the body
    # Find position of ':-'
    if ':-' not in rule_str:
        return []
        
    start_index = rule_str.find(':-') + 2
    
    # Find position of the last '.'
    end_index = rule_str.rfind('.')
    
    if end_index == -1:
        # If no trailing dot, take the rest of the string
        body_str = rule_str[start_index:]
    else:
        body_str = rule_str[start_index:end_index]
        
    body_str = body_str.strip()
    
    # 2. Split body into goals by top-level commas
    goals = []
    current_goal = []
    paren_depth = 0
    in_quote = False
    quote_char = None
    
    for char in body_str:
        if in_quote:
            current_goal.append(char)
            # If we hit the matching quote, close the string
            if char == quote_char:
                in_quote = False
                quote_char = None
        else:
            if char == "'" or char == '"':
                in_quote = True
                quote_char = char
                current_goal.append(char)
            elif char == '(':
                paren_depth += 1
                current_goal.append(char)
            elif char == ')':
                paren_depth -= 1
                current_goal.append(char)
            elif char == ',' and paren_depth == 0:
                # Top-level comma found; push the goal
                goals.append("".join(current_goal).strip())
                current_goal = []
            else:
                current_goal.append(char)
                
    # Append the final goal if exists
    if current_goal:
        goals.append("".join(current_goal).strip())
        
    # 3. Classify each goal
    classified_goals = []
    
    for goal in goals:
        if not goal:
            continue
            
        g = goal.strip()
        goal_type = 'predicate' # Default
        
        # Check types in order of specificity
        
        # Cut
        if g == '!':
            goal_type = 'cut'
            
        # Negation
        elif g.startswith(r'\+'):
            goal_type = 'negation'
            
        # Disjunction (contains semicolon)
        elif ';' in g:
            # Note: We assume the semicolon is an operator, not part of a quoted string.
            # (Simplification per prompt instructions)
            goal_type = 'disjunction'
            
        # Arithmetic IS (check for " is " surrounded by whitespace or boundaries)
        elif re.search(r'\bis\b', g):
            goal_type = 'arithmetic_is'
            
        # Comparison
        # Look for >, <, >=, =<, =:=, =\=
        elif re.search(r'(>=|=<|=:=|=\\=|>|<)', g):
            goal_type = 'comparison'
            
        # Unification
        # Look for = but ensure it's not part of ==, =<, >=, etc.
        # Regex: Look for = that is NOT preceded by <, >, \, ! and NOT followed by <, >, =
        elif re.search(r'(?<![<>\\!])=(?![<>=])', g):
            goal_type = 'unification'
            
        # If none of the above match, it remains 'predicate'
        
        classified_goals.append((goal_type, g))
        
    return classified_goals

def get_classification_rule_body_premises_df(df):
    rule_premises_records = []

    rules_info_df, rules_info = get_rules_info(df)

    for _, row in rules_info_df.iterrows():
        question_index = row['question_index']
        model = row['model']
        rule = row['rule']
        head = row['head']
        body = row['body']
        body_len = row['body_len']

        premises = classify_rule_body_premises(rule)
        for premise_type, premise in premises:
            rule_premises_records.append({
                "question_index": question_index,
                "model": model,
                "rule": rule,
                "head": head,
                "body": body,
                "body_len": body_len,
                "premise_type": premise_type,
                "premise": premise
            })

    rule_premises_df = pd.DataFrame(rule_premises_records)
    return rule_premises_df


def get_df_facts_usage(df):
    rf_records = []
    for _, row in df.iterrows():
        question_index = row['question_index']
        reasoning = row['reasoning']
        model = row['model']

        clauses = split_prolog_clauses(reasoning)

        facts = set()
        rules_bodies = []

        for c in clauses:
            if is_rule(c):
                try:
                    head, body = parse_rule(c)
                    body_preds = [signature(b) for b in body]
                    rules_bodies.extend(body_preds)
                except ValueError as e:
                    pass
            else:
                facts.add(signature(c))

        rules_bodies = set(rules_bodies)

        used_facts = facts.intersection(rules_bodies)
        unused_facts = facts.difference(rules_bodies)
        used_but_not_defined = rules_bodies.difference(facts)

        rf_records.append({
            "question_index": question_index,
            "model": model,
            "n_facts": len(facts),
            "n_used_facts": len(used_facts),
            "n_unused_facts": len(unused_facts),
            "n_predicates_used_but_not_defined": len(used_but_not_defined),
            "facts": sorted(facts),
            "used_facts": sorted(used_facts),
            "unused_facts": sorted(unused_facts),
            "predicates_used_but_not_defined_as_fact": sorted(used_but_not_defined)
        })
    
    df_facts_usage = pd.DataFrame(rf_records)
    return df_facts_usage

## SIMILARITIES

def jaccard(a, b):
    a, b = set(a), set(b)
    union = a | b
    if len(union) == 0:
        return 0
    return len(a & b) / len(a | b)