def build_classification_prompt(prompt_dict, options, add_generation_prompt=True):
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

