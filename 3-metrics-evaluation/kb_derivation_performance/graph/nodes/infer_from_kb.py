from graph.state import InferenceState
from pyswip import Prolog
from config import KB_PATH

def infer_from_kb_node(state: InferenceState):
    """
    Infers the conclusion(s) from the KB with a timeout.
    """

    prolog = Prolog()
    prolog.consult(KB_PATH)

    try:
        # 5-second timeout
        query_str = "call_with_time_limit(60, diagnosis(X))" # ? we put 60 seconds as the timeout.
        results = prolog.query(query_str)

        state['kb_conclusions'] = [str(x['X']) for x in results]
        return state

    except Exception as e:
        error_str = (
            f"Exception encountered when inferring from the KB. "
            f"Index={state['question_index']}. Error={e}"
        )
        print(error_str)

        state['kb_conclusions'] = []
        state['errors'].append(error_str)
        return state