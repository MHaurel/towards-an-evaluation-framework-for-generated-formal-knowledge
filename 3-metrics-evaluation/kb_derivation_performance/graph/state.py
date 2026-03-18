from typing import TypedDict, List, Optional


class InferenceState(TypedDict):
    # The index of the problem
    question_index: int

    # The text given as the input
    input_text: str

    # The name of the KB
    kb_filename: Optional[str]

    # The current KB
    kb: Optional[str]

    # Extracted facts
    extracted_facts: Optional[str]

    # Temporary KB name
    temp_kb_filename: Optional[str]

    # The conclusion of the KB
    kb_conclusions: Optional[List[str]]

    # The targeted conclusion under a NL format
    target_nl_conclusion: Optional[str]

    # The errors encountered on the way
    errors: List[str]