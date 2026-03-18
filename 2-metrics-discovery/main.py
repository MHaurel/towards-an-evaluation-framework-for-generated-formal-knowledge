import json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from jinja2 import FileSystemLoader, Environment
from dotenv import load_dotenv
import os
from pathlib import Path
import datetime
from pydantic import BaseModel
from typing import List
from tqdm import tqdm
import re

def extract_json_block(text):
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else text

here = Path(__file__).parent.resolve()

load_dotenv(os.path.join(here, '.env'))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "anthropic/claude-sonnet-4.6"
# DEBUG_MODEL = "google/gemini-2.5-flash-lite"
TEMPERATURE = 0

class OutputSchema(BaseModel):
    fix_name: str
    fix_description: str
    reason: str

DATASET_FILEPATH = "results_prolog-computed.csv" # using the prolog computed with arity explicitly demanded.

# Setup Jinja2 environment
jinja_env = Environment(loader=FileSystemLoader(here / "prompts"))
system_template = jinja_env.get_template("system.j2")
user_template = jinja_env.get_template("user.j2")

# Setup LLM
llm = ChatOpenAI(
    # model=DEBUG_MODEL,
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=TEMPERATURE
)

df = pd.read_csv(DATASET_FILEPATH)

rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    prolog_code = row['reasoning']
    problem = row['question']
    question_index = row['question_index']

    system_prompt = system_template.render()
    user_prompt = user_template.render(problem=problem, prolog_code=prolog_code)

    messages = [
        ("system", system_prompt),
        ("user", user_prompt)
    ]

    response = llm.invoke(messages)
    content = response.content

    try:
        content_json = extract_json_block(content)
        fixes: List[dict] = json.loads(content_json)
    except json.JSONDecodeError as e:
        print(f"JSON Decode error: {e}")
        fixes = []

    for fix in fixes:
        rows.append({
            "question_index": question_index,
            "question": problem,
            "reasoning": prolog_code,
            "fix_name": fix.get("fix_name", ""),
            "fix_description": fix.get("fix_description", ""),
            "reason": fix.get("reason", ""),
        })

df_results = pd.DataFrame(rows)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = here / f"results_{timestamp}_arity.csv"
df_results.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
