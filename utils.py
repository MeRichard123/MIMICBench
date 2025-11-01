from typing import TypedDict, List
import random

Prompt = TypedDict("Prompt", {
    "prompt_id": str,
    "prompt": str,
    "ground_truth": str,
    "note": str
})

def build_classification_prompt(prompt_dict: Prompt, options, add_generation_prompt=True):
    """Build prompt - define this ONCE outside the function"""
    prompt_text = prompt_dict["prompt"]
    parts = []

    system_message = (
        "You are a medical coding assistant. Your task is to assign the most appropriate ICD-9 code to the medical note below."
        "Read the note carefully and respond with only one ICD-9 code from the list."
    )
    parts.append(f"[system] {system_message}")
    parts.append(f"[user]")
    parts.append(f"Possible ICD-9 codes:: \n")
    for k,v in options.items():
        parts.append(f"{k}: {v}")
    parts.append(f"Medical Note: \n")
    parts.append("------------------------")
    parts.append(prompt_text)
    parts.append("------------------------")
    parts.append("Respond with only the ICD-9 code. Do not include the title or explanation.")


    prompt = "\n".join(parts)

    if add_generation_prompt:
        prompt += "\n[assistant]"

    return prompt

def get_random_choices(correct: str, options: Prompt) -> List[str]:
    choices = []
    while correct in choices or len(choices) == 0:
        choices = random.sample(options, k=3) 
    choices.append(correct)
    random.shuffle(choices)
    return choices

questions = [
    "According to the medical note, what ICD9 code is relevant?",
    "What ICD9 code would you assign for the information in the medical note?",
    "Identify the relevant ICD9 code as derived from the medical note.",
    "What ICD9 diagnostic code can be inferred from the medical note?",
    "Which ICD9 code is indicated by the information in this medical note?",
    "From this medical note, which ICD9 code is recommended?",
    "What ICD9 code corresponds to this medical note?",
    "Which ICD9 diagnostic code applies to the medical note provided?",
    "Determine the ICD9 code that matches the content of the medical note.",
    "From the medical note, what is the applicable ICD9 code?",
    "What is the correct ICD9 code based on the details in the medical note?",
    "Given the medical note, which ICD9 code should be assigned?"
]

def build_QA_prompt(prompt_dict: Prompt, options):
    note = prompt_dict["prompt"]
    correct_option = prompt_dict["ground_truth"]
    choices = get_random_choices(correct_option, list(options.keys()))

    parts = []

    system_message = (
        "You are an expert medical coding assistant. Please provide the correct ICD9 code based "
        "on the medical note and given options."
    )

    parts.append(f"[system] {system_message}")
    parts.append("[user]")
    parts.append(f"Medical Note: \n")
    parts.append("------------------------")
    parts.append(note)
    parts.append("------------------------")

    parts.append(random.choice(questions))
    for i, choice in enumerate(choices):
        parts.append(f"{chr(ord('a') + i)}) {choice}: {options[choice]}")
    
    prompt = "\n".join(parts)
    return prompt
    
