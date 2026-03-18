from jinja2 import Environment, FileSystemLoader
import os
from pathlib import Path

here = Path(__file__).parent.resolve()

def build_assess_kb_response_prompt(kb_response: str, target_nl_conclusion: str):
    env = Environment(
        loader=FileSystemLoader(os.path.join(here, "prompts", "assess_kb_response")),
        trim_blocks=True,
        lstrip_blocks=True
    )

    user_template = env.get_template('user.j2')
    user_prompt = user_template.render(
        kb_response=kb_response,
        target_nl_conclusion=target_nl_conclusion
    )

    system_template = env.get_template('system.j2')
    SYSTEM_PROMPT = system_template.render()

    return SYSTEM_PROMPT, user_prompt