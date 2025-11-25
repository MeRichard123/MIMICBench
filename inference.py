import torch
from unsloth import FastLanguageModel
import os
import gc
import pandas as pd
import matplotlib.pyplot as plt
from utils import build_classification_prompt, build_QA_prompt, build_open_QA_prompt, load_latest_checkpoint
import json
from tqdm import tqdm
from dotenv import load_dotenv
import transformers
from enum import Enum

class Tasks(Enum):
    OQA = 'oqa'
    CLASS = 'classification'
    QA = 'qa'

load_dotenv()

# unsloth/Meta-Llama-3.1-8B-bnb-4bit
# unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
# google/medgemma-4b-it
# Kavyaah/medical-coding-llm
# epfl-llm/meditron-7b
# haohao12/qwen2.5-7b-medical

TASK = Tasks.CLASS
MODEL = "meditron-7b"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "epfl-llm/meditron-7b",
    dtype = None,
    max_seq_length = 4096,
    load_in_4bit = True,
    full_finetuning = False,
    device_map = "auto",
    low_cpu_mem_usage = True,
    offload_folder = os.path.join(os.getcwd(), "offload"),
    token=os.getenv("HF_TOKEN"),
)

# disable checkpointing for models that don't support it 
model.config.use_cache = True
model.config.gradient_checkpointing = False
if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()

model.eval()  # inference mode

if MODEL == "medgemma-4b-it":
    # medgemma tokenizer from unsloth is broken so re-load from transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "google/medgemma-4b-it",
        use_fast=True,
        token=os.getenv("HF_TOKEN")
    )

FastLanguageModel.for_inference(model)


def run_inference(prompt_dict, options):
    """Run inference using the pre-loaded model"""
    logging = False

    # Get device from model (don't reassign device)
    device = next(model.parameters()).device

    # Build prompt defensively

    if TASK == Tasks.CLASS:
        prompt = build_classification_prompt(prompt_dict, options, add_generation_prompt=True)
    elif TASK ==  Tasks.QA:
        prompt = build_QA_prompt(prompt_dict, options)
    elif TASK == Tasks.OQA:
        prompt = build_open_QA_prompt(prompt_dict, options)
    else:
        raise ValueError(f"Unknown TASK: {TASK}")

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
            forward_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10, 
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.0,
            )



        logits = forward_out.logits
        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)

        generated_tokens = outputs[0][input_ids.shape[1]:]
        result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


        code_token_ids = {
            code: tokenizer(code, add_special_tokens=False).input_ids
            for code in options
        }      

            # Map ICD9 codes → token probs
        code_probs = {}
        for code, ids in code_token_ids.items():
            code_probs[code] = probs[ids[0]].item()

            # # Normalize
        total = sum(code_probs.values())
        for c in code_probs:
            code_probs[c] /= total


        if TASK != Tasks.OQA:
            # Clean result - remove any assistant tags or extra text
            result = result.replace('[assistant]', '').strip()
            if '[system]' in result:
                result = result.split('[system]')[0].strip()

            result = result.split('\n')[0].split(';')[0].strip()  # Take first part

        if logging:
            print(f"Predicted: {result}")
            print(f"Probability Slice {code_probs}")
            print(f"Actual: {prompt_dict['ground_truth']}")
            print("-" * 50)

        return result, code_probs

    except Exception as e:
        print(f"Model generation failed: {e}")
        return None


df = pd.read_csv("/workspaces/CMP9794-Advanced-Artificial-Intelligence/MIMIC-SAMPLED.csv")

null_rows = df[df.isnull().any(axis=1)]
print(null_rows.shape)
df.isnull().sum()

df_copy = df.copy()
df_copy.dropna(subset=["ICD9 Diagnosis"], inplace=True)
df_copy.dropna(subset=["note_category"], inplace=True)
df_copy.isnull().sum()

# Calculate the number of rows in the original and modified dataframes
original_rows = df.shape[0]
dropped_rows = original_rows - df_copy.shape[0]
not_dropped_rows = df_copy.shape[0]

# Create labels and sizes for the pie chart
labels = ['Dropped Rows', 'Not Dropped Rows']
sizes = [dropped_rows, not_dropped_rows]
colors = ['#ff9999','#66b3da']
explode = (0.1, 0)  # explode 1st slice

# Create the pie chart
plt.figure(figsize=(5, 5))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Proportion of Dropped vs Not Dropped Rows')
plt.savefig("DroppedPie.png")


df_sampled = df_copy.sample(n=1312, random_state=42) # Shuffle
print(df_sampled.head())

unique_pairs = df[['ICD9 Diagnosis', 'SHORT_TITLE']].drop_duplicates()
code_to_title = dict(unique_pairs.values)


df_sampled = df_sampled.reset_index()
classification_prompts = []
labels = []  # For ground truth

for i, sample in df_sampled.iterrows():
    diagnosis = sample["ICD9 Diagnosis"]
    note = sample["Note"]

    classification_prompts.append({
        "prompt_id": f"classify_{i}",
        "prompt": note,
        "ground_truth": diagnosis,
        "note": note
    })
    labels.append(diagnosis)

results = []


try:
    checkpoint = load_latest_checkpoint(MODEL, TASK.value)
    offset = 0
    if checkpoint:
        file = open(checkpoint, "r")
        json_contents = json.load(file)
        print(f"LOADED CHECKPOINT {file.name}")
        results.extend(json_contents)
        offset = len(json_contents)

    for i in tqdm(range(len(classification_prompts) - offset)):
        prompt = classification_prompts[i]
        obj = {}
        inference, probs = run_inference(prompt, code_to_title)
        obj["predicted_diagnosis"] = inference
        obj["ground_truth"] = prompt["ground_truth"]
        obj['probabilities'] = probs
        results.append(obj)

    with open(f"{MODEL}-{TASK.value}.json", "w") as f:
        json.dump(results, f)
except KeyboardInterrupt:
    import datetime
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    with open(f"{MODEL}-{TASK.value}-checkpoint-{timestamp}.json", "w") as f:
        json.dump(results, f)