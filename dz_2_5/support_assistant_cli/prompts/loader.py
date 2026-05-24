import json
from importlib import resources
from typing import Any
from jinja2 import Template
from sympy import content


def _read_prompt_file(filename:str)->str:
    return resources.files(__package__).joinpath(filename).read_text(encoding="utf-8").strip()

SYSTEM_PROMPT_TEMPLATE = _read_prompt_file("system_prompt.txt")
CLASSIFIER_SYSTEM_PROMPT = _read_prompt_file("classifier_system_prompt.txt")
CLASSIFIER_FEW_SHOTS = json.loads(_read_prompt_file("classifier_few_shots.json"))
ANSWERS_FEW_SHOTS = json.loads(_read_prompt_file("answers_few_shots.json"))

def build_system_prompt()->str:
    return Template(SYSTEM_PROMPT_TEMPLATE).render(
        answers_few_shots=ANSWERS_FEW_SHOTS
    )

def build_classifier_system_prompt()->str:
    return Template(CLASSIFIER_SYSTEM_PROMPT).render(
        classifier_few_shots=CLASSIFIER_FEW_SHOTS
    )

def build_answer_messages(system_prompt:str, history: list[dict[str,str]], user_message:str) -> list[dict[str,str]]:
    messages: list[dict[str,str]] = [{'role':'system', 'content':system_prompt}]
    messages.extend(history)
    messages.append({'role':'user', 'content':user_message})
    return messages

def build_classifier_messages(system_prompt:str, user_message:str) -> list[dict[str,str]]:
    messages: list[dict[str,Any]] = [{'role':'system', 'content':system_prompt}]
    messages.append({'role':'user', 'content':user_message})
    return messages
