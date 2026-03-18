from langgraph.graph import StateGraph, END, START

from graph.state import InferenceState

from graph.nodes.write_to_kb import write_to_kb_node
from graph.nodes.infer_from_kb import infer_from_kb_node

def create_graph() -> StateGraph:
    """
    Creates and compiles the graph that takes text as input and infer the conclusion from the KB.
    """
    workflow = StateGraph(InferenceState)

    workflow.add_node("write_to_kb", write_to_kb_node)
    workflow.add_node("infer_from_kb", infer_from_kb_node)
    
    workflow.set_entry_point("write_to_kb")

    workflow.add_edge(START, "write_to_kb")
    workflow.add_edge("write_to_kb", "infer_from_kb")
    workflow.add_edge("infer_from_kb", END)

    graph = workflow.compile()
    return graph