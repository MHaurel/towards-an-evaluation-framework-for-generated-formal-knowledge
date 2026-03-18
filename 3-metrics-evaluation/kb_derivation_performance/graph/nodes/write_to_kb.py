from graph.state import InferenceState

from config import KB_PATH

def write_to_kb_node(state: InferenceState) -> InferenceState:
    """
    Loads the Prolog KB.
    """
    # print("--- load_kb_node ---")

    # prolog_kb = None
    with open(KB_PATH, 'w') as f:
        f.write(state['kb'])

    # state is already loaded, we dismiss this step.
    # This will be useful when incrementally building the KB because we need to pass it as the context. For now, we ignore it.

    return state