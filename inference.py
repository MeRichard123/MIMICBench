import torch
from unsloth import FastLanguageModel
import os
import gc
import pandas as pd
import matplotlib.pyplot as plt
from utils import build_classification_prompt
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# google/medgemma-4b-it
# Kavyaah/medical-coding-llm
# unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
# haohao12/qwen2.5-7b-medical

TASK = "classification"
MODEL = "Qwen2.5-7b-Medical"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "haohao12/qwen2.5-7b-medical",
    dtype = None,
    max_seq_length = 4096,
    load_in_4bit = True,
    full_finetuning = False,
     # let accelerate/transformers decide placement and use low_cpu_mem_usage
    device_map = "auto",
    low_cpu_mem_usage = True,
    offload_folder = os.path.join(os.getcwd(), "offload"),
    token=os.getenv("HF_TOKEN")
)

FastLanguageModel.for_inference(model)


def run_inference(prompt_dict, options):
    """Run inference using the pre-loaded model"""
    logging = False

    # Get device from model (don't reassign device)
    device = next(model.parameters()).device

    if TASK == "classification":
        prompt = build_classification_prompt(prompt_dict, options, add_generation_prompt=True)
    else:
        raise ValueError(f"Unknown TASK: {TASK}")

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
                max_new_tokens=10, 
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.0,
            )

        # Decode only new tokens
        generated_tokens = outputs[0][input_ids.shape[1]:]
        result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Clean result - remove any assistant tags or extra text
        result = result.replace('[assistant]', '').strip()
        if '[system]' in result:
            result = result.split('[system]')[0].strip()

        result = result.split('\n')[0].split(';')[0].strip()  # Take first part

        if logging:
          print(f"Predicted: {result}")
          print(f"Actual: {prompt_dict['ground_truth']}")
          print("-" * 50)

        return result

    except Exception as e:
        print(f"Model generation failed: {e}")
        return None

def cleanup_memory():
    """Call this between batches if needed"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


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


for i in tqdm(range(len(classification_prompts))):
    prompt = classification_prompts[i]
    obj = {}
    inference = run_inference(prompt, code_to_title)
    obj["predicted_diagnosis"] = inference
    obj["ground_truth"] = prompt["ground_truth"]
    results.append(obj)

with open(f"{MODEL}-{TASK}.json", "w") as f:
    json.dump(results, f)