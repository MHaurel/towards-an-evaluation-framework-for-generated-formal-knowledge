import pandas as pd
import re
from llama_index.llms.ollama import Ollama
from tqdm import tqdm
from datetime import datetime
import traceback
import time
import os
from jinja2 import Environment, FileSystemLoader

MAX_RETRIES = 3
RETRY_DELAY = 5

env = Environment(
    loader=FileSystemLoader("prompts"),
    trim_blocks=True,
    lstrip_blocks=True
)

def extract_reasoning(text):
    match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    return "", text

def log_error(error_message, prompt=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("error_log.txt", "a") as f:
        f.write(f"[{timestamp}] {error_message}\n")
        if prompt:
            f.write(f"Prompt:\n{prompt}\n")
        f.write(traceback.format_exc())
        f.write("\n\n")

def save_results_batch(results, batch_num, model_name, timestamp):
    if not results: return
    df_batch = pd.DataFrame(results)
    os.makedirs("outputs", exist_ok=True)
    filename = f"outputs/results_{timestamp}_batch_{batch_num:03d}_{model_name.replace(':', '_')}.csv"
    df_batch.to_csv(filename, index=False)
    print(f"✅ Batch {batch_num} saved to {filename} ({len(results)} problems)")

def load_existing_results(timestamp, model_name):
    results_dir = "outputs"
    if not os.path.exists(results_dir): return []
    existing_results = []
    model_safe_name = model_name.replace(':', '_')
    batch_files = [f for f in os.listdir(results_dir) if f.startswith(f"results_{timestamp}") and model_safe_name in f]
    for batch_file in sorted(batch_files):
        try:
            df_batch = pd.read_csv(os.path.join(results_dir, batch_file))
            existing_results.extend(df_batch.to_dict('records'))
        except Exception as e:
            print(f"⚠️ Could not load {batch_file}: {e}")
    return existing_results

def main():
    TIMEOUT = 600
    BATCH_SIZE = 50 

    try:
        df = pd.read_csv('dataset_diagnosis_classified.csv')
        
        df = df[df['is_diagnosis'] == 1]

        N_PROBLEMS = 100
        TEMPERATURE = 0

        print(f"{df.shape} diagnosis problems in total. Using {N_PROBLEMS}.")

        df = df.iloc[:N_PROBLEMS]

        llms = [
            Ollama(model="cogito:14b", request_timeout=TIMEOUT, temperature=TEMPERATURE),
        ]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        all_results = []

        for llm in tqdm(llms, desc="LLMs"):
            existing_results = load_existing_results(timestamp, llm.model)
            processed_count = len(existing_results)
            all_results.extend(existing_results)
            
            current_batch = []
            batch_num = (processed_count // BATCH_SIZE) + 1
            start_idx = processed_count
            
            for i, row in tqdm(df.iloc[start_idx:].iterrows(), total=len(df)-start_idx, desc=f"Running {llm.model}", leave=False):
                problem = row['question']

                template = env.get_template("prompt.j2")

                prompt = template.render(user_input=problem)

                response, reasoning, answer = None, "", ""
                duration = None
                success = False
                
                for attempt in range(MAX_RETRIES):
                    try:
                        start_time = time.time()
                        response = llm.complete(prompt)
                        duration = time.time() - start_time
                        reasoning, answer = extract_reasoning(response.text)
                        success = True
                        break # Success! Exit the retry loop
                    except Exception as e:
                        wait_time = RETRY_DELAY * (attempt + 1)
                        log_error(f"Attempt {attempt+1} failed — Model: {llm.model}, Row: {i}. Error: {str(e)}")
                        if attempt < MAX_RETRIES - 1:
                            print(f"⚠️ Attempt {attempt+1} failed for {llm.model}. Error: {str(e)}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"❌ All {MAX_RETRIES} attempts failed for model {llm.model} on question {i}")

                result = {
                    "model": llm.model,
                    "prompt": prompt,
                    "question": problem,
                    "raw_output": response.text if response else "ERROR_OR_TIMEOUT", 
                    "reasoning": reasoning,
                    "answer": answer,
                    "latency_sec": round(duration, 2) if duration else None,
                    "question_index": i,
                    "processed_at": datetime.now().isoformat(),
                    "status": "success" if success else "failed"
                }
                
                current_batch.append(result)
                all_results.append(result)

                if len(current_batch) >= BATCH_SIZE:
                    save_results_batch(current_batch, batch_num, llm.model, timestamp)
                    current_batch = []
                    batch_num += 1

            if current_batch:
                save_results_batch(current_batch, batch_num, llm.model, timestamp)

        df_results = pd.DataFrame(all_results)
        final_filename = f"outputs/results_{timestamp}_constraints_consolidated.csv"
        df_results.to_csv(final_filename, index=False)

    except Exception as e:
        log_error("Fatal error in main()")
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()