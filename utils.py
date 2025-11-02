from typing import TypedDict, List
import random
import math

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

def get_random_choices(correct: str, options) -> List[str]:
    """Return a list of 4 choices (3 random + correct). Handles small option sets.

    - options may be any iterable of keys/choices.
    - If correct is not in options, we fall back to the first available option.
    """
    options_list = list(options)

    # If there are no options, return the correct answer as the sole choice
    if len(options_list) == 0:
        return [correct]

    # Ensure correct is a valid choice
    if correct not in options_list:
        correct = options_list[0]

    choices = []
    # If fewer than 3 available options, sample with replacement to build 3
    if len(options_list) < 3:
        while len(choices) < 3:
            c = random.choice(options_list)
            if c not in choices:
                choices.append(c)
            else:
                # allow duplicates only if we can't fill unique ones
                if len(set(choices)) == len(options_list):
                    choices.append(c)
    else:
        choices = random.sample(options_list, k=3)

    if correct not in choices:
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
    # Guard against missing/NaN notes and ground truth values
    note = prompt_dict.get("prompt", "")
    correct_option = prompt_dict.get("ground_truth", None)

    try:
        if note is None or (isinstance(note, float) and math.isnan(note)):
            note = ""
    except Exception:
        pass

    note = str(note)

    option_keys = list(options.keys())
    if correct_option is None or (isinstance(correct_option, float) and math.isnan(correct_option)):
        correct_option = option_keys[0] if option_keys else ""

    choices = get_random_choices(correct_option, option_keys)

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
    
