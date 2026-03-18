from graph.graph import create_graph
from graph.state import InferenceState
from utils import load_results
import pandas as pd
from tqdm import tqdm
import datetime

def get_response(df_base, index):
    return df_base.iloc[index]['Response']

def get_results(save=False):

    rows_full_results = []

    df_results = load_results(
        path="../3-metrics-evaluation/results_prolog-computed.csv",
        only_perfect=True, # we choose only perfect programs since they need to be loaded successfully in the interpreter
        enforce=False
    )

    df_base = pd.read_json("hf://datasets/FreedomIntelligence/medical-o1-reasoning-SFT/medical_o1_sft.json")
    
    for index, row in tqdm(df_results.iterrows(), total=len(df_results)):

        input_text = row['question']
        reasoning = row['reasoning']
        question_index = row['question_index']
        target_nl_conclusion = get_response(df_base=df_base, index=index)

        graph = create_graph()

        initial_state: InferenceState = {
            "question_index": int(question_index),
            "input_text": input_text,
            "kb_filename": None,
            "kb": reasoning,
            "extracted_facts": None,
            "temp_kb_filename": None,
            "kb_conclusions": None,
            "target_nl_conclusion": target_nl_conclusion,
            "errors": []
        }

        result = graph.invoke(initial_state)

        rows_full_results.append(result)
    
    df_full_results = pd.DataFrame(rows_full_results)

    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        df_full_results.to_csv(f'full_results_{timestamp}.csv', index=False)

    return df_full_results

if __name__ == "__main__":
    get_results(save=True)