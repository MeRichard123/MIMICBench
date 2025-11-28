from typing import TypedDict, List
import random, os, datetime
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
        "You are an expert medical coding assistant. Please answer the questions with the correct ICD9 code based "
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

open_ended_questions = [
    "What ICD9 code best reflects the diagnosis in the medical note?",
    "Based on the details in the medical note, how would you categorize the patient's condition using an ICD9 code?",
    "Can you provide the appropriate ICD9 code that describes the situation presented in the medical note?",
    "What ICD9 code would you determine is most applicable for the information detailed in this medical note?",
    "From your analysis of the medical note, what ICD9 code would you assign, and why?",
    "Considering the information in the medical note, what would be the most suitable ICD9 code to reflect the diagnosis?",
    "In your assessment of the medical note, which ICD9 code do you think is the most appropriate, and what factors led you to that conclusion?",
    "Reflecting on the medical note, what ICD9 code could you suggest to classify the patient’s diagnosis?",
    "After reviewing the medical note, what ICD9 code comes to mind, and what elements of the note support your answer?",
    "Taking into account the details provided in the medical note, which ICD9 code would you recommend for medical coding?"
]

def build_open_QA_prompt(prompt_dict: Prompt, options):
    # Guard against missing/NaN notes and ground truth values
    note = prompt_dict.get("prompt", "")

    try:
        if note is None or (isinstance(note, float) and math.isnan(note)):
            note = ""
    except Exception:
        pass

    note = str(note)

    parts = []

    system_message = (
        "You are an expert medical coding assistant. Please provide a detailed explanation of the ICD-9 code based on the medical note."
    )

    parts.append(f"[system] {system_message}")
    parts.append("[user]")
    parts.append(f"Medical Note: \n")
    parts.append("------------------------")
    parts.append(note)
    parts.append("------------------------")

    parts.append(random.choice(open_ended_questions))

    
    prompt = "\n".join(parts)
    return prompt
    
def load_latest_checkpoint(model, task):
    checkpoint_folder = '/workspaces/CMP9794-Advanced-Artificial-Intelligence/Checkpoints'
    checkpoint_files = []

    for filename in os.listdir(checkpoint_folder):
        if filename.startswith(f"{model}-{task}-checkpoint-") and filename.endswith(".json"):
            file_path = os.path.join(checkpoint_folder, filename)
            timestamp_str = filename[len(f"{model}-{task}-checkpoint-"):-len(".json")]
            try:
                # Adjust the format to match the way we create filenames
                timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
                checkpoint_files.append((file_path, timestamp))
            except ValueError:
                continue  # Skip files that don't match the expected format

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])[0]
        return latest_checkpoint
    else:
        return None
    


class ReferenceImplementation(Exception):
    def __init__(self, message):
        super().__init__(message)


def reference_impl(func):
    def inner(*vaargs):
        raise ReferenceImplementation("This is is a reference implemetation serving as documentation for a function \n that was defined in a different way. This should not be called.")
    return inner