def build_prompt(prompt_dict, options, add_generation_prompt=True):
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



def build_eval_prompt(prediction, ground_truth, options):
    """
    This function will evaluate the predicted diagnosis against the ground truth.
    It prompts an LLM to act as a judge to decide the correctness and quality of the prediction.
    """
    parts = []

    # Define the evaluation task and prompt for the LLM
    parts.append("You are an expert medical coding assistant and your task is to evaluate the quality of ICD-9 code predictions.")

    parts.append(f"Possible ICD-9 codes: \n")
    for k,v in options.items():
        parts.append(f"{k}: {v}")
    
    parts.append(f"""
    Here is the prediction and the ground truth for a medical case:

    Predicted Diagnosis: {prediction}
    Ground Truth Diagnosis: {ground_truth}
    """)


    parts.append("""
    Task: 
    1. Check if the predicted diagnosis (ICD-9 code) matches the ground truth diagnosis code.
    2. Evaluate whether the predicted ICD-9 code is appropriate for the medical note (given the context).
    3. Consider the precision of the diagnosis and whether the generated code accurately reflects the condition described.

    Please provide a brief response with your evaluation of the prediction:
    - Is the predicted code correct? (Yes/No)
    - If the predicted code is incorrect, what is the correct ICD-9 code?
    - Is the description relevant to the ICD-9 code? (Yes/No)
    - If the description is incorrect, provide a better description.

    Please also provide a score from 0 to 1 based on the correctness of the predicted diagnosis:
    0 means totally incorrect, 1 means completely correct.
    """)

    return '\n'.join(parts)
