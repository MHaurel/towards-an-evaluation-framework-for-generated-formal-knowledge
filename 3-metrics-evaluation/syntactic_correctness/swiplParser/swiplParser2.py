import subprocess
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def evaluate_prolog_syntax2(prolog_code: str) -> float:
    if not isinstance(prolog_code, str):
        return 0

    if prolog_code.strip() == "":
        return 0
    
    # print(prolog_code) # print the clause

    tmp_filename = None # Initialize to None for cleanup in finally block

    try:
        # write code to temp file
        with tempfile.NamedTemporaryFile(suffix=".pl", delete=False, mode="w", encoding='utf-8') as f: # Specify encoding
            f.write(prolog_code)
            tmp_filename = f.name

        # run swipl to check syntax
        # The key change: add the timeout parameter here!
        # Make this timeout shorter than the outer Future timeout if possible,
        # to allow subprocess.run to raise the exception and be handled.
        SWIPL_TIMEOUT_SECONDS = 60 # Example: Give swipl 60 seconds
        
        result = subprocess.run(
            ['swipl', '-q', '-t', 'halt', '-f', tmp_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=SWIPL_TIMEOUT_SECONDS # <-- This is the crucial addition!
        )
        
        # count error lines
        error_lines = result.stderr.strip().split('\n')
        # print("error lines:", error_lines)
        n = sum(1 for line in error_lines if "ERROR:" in line or "Syntax error" in line)

    except subprocess.TimeoutExpired:
        # swipl took too long. Handle this specifically.
        print(f"SWIPL call timed out after {SWIPL_TIMEOUT_SECONDS} seconds for code starting: {prolog_code[:50]}...")
        # You might want to log stdout/stderr if available even on timeout
        # print(f"STDOUT: {result.stdout}") # result might not be fully populated
        # print(f"STDERR: {result.stderr}")
        n = 1 # Assign a value that ensures a 0.0 reward for timeout
    except Exception as e:
        # Catch any other unexpected errors from subprocess.run or file ops
        print(f"An error occurred during SWIPL execution or file operation: {e} for code starting: {prolog_code[:50]}...")
        n = 1 # Assign a value that ensures a 0.0 reward for error
    finally:
        # cleanup, ensuring it runs even if an exception occurs
        if tmp_filename and os.path.exists(tmp_filename):
            try:
                os.remove(tmp_filename)
            except OSError as e:
                print(f"Error removing temporary file {tmp_filename}: {e}")


    # print("n", n)
    # print("m", m)
    return n
    # return 1/(n+1)

    # compute score
    if n <= 0:
        return 1.0
    else:
        # print(f"n: {n}", f"m: {m}")
        return max(0.0, 1 - (n/m)) if m > 0 else 0.0


def evaluate_single_clause2(clause):
    """Wrap a clause for evaluation — adds period if needed."""
    clause = clause.strip()
    if not clause.endswith('.'):
        clause += '.'
    return evaluate_prolog_syntax2(clause)

def evaluate_prolog_syntax_per_clause2(prolog_code, max_workers=None):
    if not isinstance(prolog_code, str):
        return 0
    
    if prolog_code.strip() == "":
        return 0
    
    prolog_code = prolog_code.strip()
    
    if not is_likely_prolog(prolog_code):
        # print("this code is not likely prolog")
        return 0

    # Fallback to per-clause evaluation
    try:
        clauses = [c.strip() for c in prolog_code.split('.') if c.strip()]
    except:
        print(type(prolog_code), prolog_code)

    # print(f"clauses length: {len(clauses)}")
    # print(f'{clauses}')

    if not clauses:
        return 0.0

    # Use all CPU cores unless specified
    max_workers = max_workers or multiprocessing.cpu_count()

    n = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all clause evaluations in parallel
        futures = [executor.submit(evaluate_single_clause2, clause) for clause in clauses]

        for future in as_completed(futures):
            try:
                n_i = future.result()
                # print("future.result:", n)
                n+=n_i
            except Exception as e:
                # Log or handle individual failures
                print(f"Error in clause evaluation: {e}")
                n_errors+=1

    # print(f"scores: {scores}")
    # print(f"scores length: {len(scores)}")

    k = len(clauses)

    # print("k:", k)
    # print("n_errors:", n)
    # print(f"1 - {n} / {k} * 100 = {1 - (n / k)}")
    return 1 - (n / k)

def is_likely_prolog(text):
    """Check if the text resembles Prolog code with more stringent criteria, ignoring comments."""
    import re
    
    # Split into lines and remove comments (lines starting with %)
    lines = text.split('\n')
    non_comment_lines = [line for line in lines if not line.strip().startswith('%')]
    
    # Check for common natural language indicators that suggest the text is NOT Prolog
    natural_language_indicators = [
        r'\b(I|me|my|we|our|us|you|your)\b',  # Personal pronouns
        r'\b(think|believe|feel|consider|understand|know|remember|figure|trying)\b',  # Cognitive verbs
        r'\b(so|because|therefore|thus|hence|since|as|given that)\b',  # Reasoning connectors
        r'\b(maybe|perhaps|possibly|probably|likely|might|could|would|should)\b',  # Modal expressions
        r'\b(first|second|third|next|then|finally|lastly)\b',  # Sequence markers
        r'\b(let me|let\'s|I\'m|I\'ll|I\'ve|I\'d)\b',  # Conversational phrases
    ]
    
    nl_indicator_count = 0
    for pattern in natural_language_indicators:
        for line in non_comment_lines:
            if re.search(pattern, line, re.IGNORECASE):
                nl_indicator_count += 1
                break  # Count each indicator only once
    
    # If we find 3 or more natural language indicators, it's almost certainly not Prolog
    if nl_indicator_count >= 3:
        return False
    
    # More specific Prolog patterns
    prolog_patterns = [
        r'\w+\([\w,\s_]+\)',  # Predicate with arguments pattern like predicate(X,Y)
        r':-\s*\w+',          # Definition operator followed by a predicate
        r'\?-\s*\w+',         # Query operator followed by a predicate
        r'\w+\s*\.',          # Fact ending with period
        r'\w+\s*:-\s*\w+',    # Rule pattern with proper spacing
    ]
    
    prolog_line_count = 0
    non_empty_lines = 0
    
    for line in non_comment_lines:
        line = line.strip()
        if line == "":
            continue
            
        non_empty_lines += 1
        
        # Check for Prolog patterns
        for pattern in prolog_patterns:
            if re.search(pattern, line):
                prolog_line_count += 1
                break
    
    if non_empty_lines == 0:
        return False
        
    # Require at least 25% of non-empty lines to contain Prolog-like syntax
    # AND at least 2 lines with Prolog syntax for short texts
    prolog_ratio = prolog_line_count / non_empty_lines
    return prolog_ratio >= 0.25 and prolog_line_count >= 2



