import os, re, json
import numpy as np
from sklearn.metrics import (
    f1_score, recall_score, accuracy_score, precision_score, 
    balanced_accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from utils import build_eval_prompt
from unsloth import FastLanguageModel
import torch
import pandas as pd


BASE_DIR = "/workspaces/CMP9794-Advanced-Artificial-Intelligence/Results/"
MODEL = "MedGemma"

df = pd.read_csv("/workspaces/CMP9794-Advanced-Artificial-Intelligence/MIMIC-SAMPLED.csv")
null_rows = df[df.isnull().any(axis=1)]
df.isnull().sum()
df_copy = df.copy()
df_copy.dropna(subset=["ICD9 Diagnosis"], inplace=True)
df_copy.dropna(subset=["note_category"], inplace=True)


unique_pairs = df_copy[['ICD9 Diagnosis', 'SHORT_TITLE']].drop_duplicates()
code_to_title = dict(unique_pairs.values)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "openai/gpt-oss-20b",
    dtype = None,
    max_seq_length = 4096,
    load_in_4bit = True,
    device_map="auto",  # Let HF automatically distribute between GPU/CPU
    offload_folder=os.path.join(os.getcwd(), "offload"),
    full_finetuning=False,
)

FastLanguageModel.for_inference(model)

def LLM_as_Judge(y_pred, ground):
    """Run inference using the pre-loaded model"""
    # Get device from model (don't reassign device)
    device = next(model.parameters()).device
    prompt = build_eval_prompt(y_pred, ground, code_to_title)
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )

    # Move to same device as model
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    try:
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256, 
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.0,
            )

        # Decode only new tokens
        generated_tokens = outputs[0][input_ids.shape[1]:]
        result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return result

    except Exception as e:
        print(f"Model generation failed: {e}")
        return None


def extract_icd9_code(result):
    if "Code:" in result:
        result = result.split("Code:")[-1]
    match = re.search(r"\b[A-Z]\d{1,3}\.?[A-Z0-9]*\b", result)
    return match.group(0).strip() if match else result.strip()



with open(os.path.join(BASE_DIR, MODEL + ".json")) as fp:
    parse = json.load(fp)

    predictions = np.array([data['predicted_diagnosis'] for data in parse])

    ground_truth = np.array([data['ground_truth'].strip() for data in parse])
    predicted = [p if p else "N/A" for p in predictions]

    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='weighted', zero_division=1)
    recall = recall_score(ground_truth, predictions, average='weighted', zero_division=1)
    f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=1)
    balanced_accuracy = balanced_accuracy_score(ground_truth, predictions)

    classes = sorted(set(ground_truth))
    cm = confusion_matrix(ground_truth, predicted, labels=classes)

    scores = []
    for pred, ground in tqdm(zip(predicted, ground_truth)):
        if pred == "N/A":
            continue
        llm_as_judge = LLM_as_Judge(pred, ground)
        match = re.search(r"Score\s*:\s*(\d+)", llm_as_judge)
        if match:
            score = int(match.group(1))
            scores.append(score)
        else:
            scores.append(0)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Average LLM Judge Score: {np.mean(scores):.4f}")
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Diagnosis')
    plt.ylabel('Ground Truth Diagnosis')
    plt.title('Confusion Matrix')
    plt.savefig(f"confmatrix-{MODEL}.png")

