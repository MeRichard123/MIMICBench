import torch
from unsloth import FastLanguageModel
import os
from utils import build_classification_prompt, build_MCQA_prompt, build_open_QA_prompt, build_QA_prompt, load_latest_checkpoint
import json
from tqdm import tqdm
from dotenv import load_dotenv
import transformers
from enum import Enum
from DataLoader import DataLoader, Dataset_t
import numpy as np

class Tasks(Enum):
    OQA = 'oqa'
    CLASS = 'classification'
    MCQA = 'mcqa'
    QA = 'qa'

load_dotenv()

# unsloth/Meta-Llama-3.1-8B-bnb-4bit
# unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
# google/medgemma-4b-it
# Kavyaah/medical-coding-llm
# epfl-llm/meditron-7b
# haohao12/qwen2.5-7b-medical

TASK = Tasks.QA
MODEL = "medical-coding-llm"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Kavyaah/medical-coding-llm",
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
    elif TASK ==  Tasks.MCQA:
        prompt = build_MCQA_prompt(prompt_dict, options)
    elif TASK == Tasks.OQA:
        prompt = build_open_QA_prompt(prompt_dict, options)
    elif TASK == Tasks.QA:
        prompt = build_QA_prompt(prompt_dict)
    else:
        raise ValueError(f"Unknown TASK: {TASK}")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=2048
    )

    # Move to same device as model
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs.get('attention_mask')
    if attention_mask is not None:  
        attention_mask = attention_mask.to(device)

    predicted_text = ""
    probabilities = {}
    generated_logprob = None
    ground_truth_logprob = None
    generated_token_count = 0

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
                return_dict_in_generate=True,
                output_scores = True
            )

            generated_ids = outputs.sequences[0, input_ids.shape[1]:]
            generated_token_count = len(generated_ids)

            if generated_token_count == 0:
                predicted_text = ''
                generated_logprob = -100.0
            else:
                predicted_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                transition_scores = model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    normalize_logits=True
                )

                gen_scores = transition_scores[0, :generated_token_count]
                generated_logprob = float(gen_scores.sum().item())

            if TASK in [Tasks.CLASS, Tasks.MCQA]:
                predicted_text = predicted_text.split('\n')[0].split(';')[0].strip()
        
        if TASK in [Tasks.CLASS, Tasks.MCQA]:
            if generated_token_count == 0:
                probabilities = {opt: 1.0 / len(options) for opt in options}
            else: 
                first_logits = outputs.scores[0][0]
                first_probs = torch.softmax(first_logits, dim=-1)

                prob_dict = {}
                for opt in options:
                    token_ids = tokenizer(opt, add_special_tokens=False).input_ids
                    if token_ids:
                        prob_dict[opt] = first_probs[token_ids[0]].item()
                    else:
                        prob_dict[opt] = 0.0
                total = sum(prob_dict.values())
                probabilities = {
                    k: v / total if total >0 else 1 / len(options)
                    for k, v in prob_dict.items()
                }

        elif TASK in [Tasks.QA, Tasks.OQA]:
            ground_truth = prompt_dict.get("ground_truth")

            if ground_truth:
                gt_ids = tokenizer(ground_truth, add_special_tokens=False).input_ids
                if len(gt_ids) > 0:
                    forced_ids = torch.cat([input_ids, ], dim=-1)
                    gt_tensor = torch.tensor(gt_ids, device=device).unsqueeze(0)
                    forced_ids = torch.cat([forced_ids, gt_tensor], dim=1)

                    with torch.no_grad():
                        out = model(forced_ids)
                        logits = out.logits[0, input_ids.shape[1]-1 : input_ids.shape[1]-1 + len(gt_ids), :]
                        log_probs = torch.log_softmax(logits, dim=-1)

                        gt_logprob = 0.0
                        for i, token_id in enumerate(gt_ids):
                            gt_logprob += log_probs[i, token_id].item()
                        ground_truth_logprob = float(gt_logprob)
                else:
                    ground_truth_logprob = 0.0
            else:
                ground_truth_logprob = None
            probabilities = {
                "type": "open_generation",
                "predicted_text": predicted_text,
                "predicted_logprob": generated_logprob if generated_logprob is not None else -100.0,
                "predicted_perplexity": (
                    np.exp(-generated_logprob / max(generated_token_count, 1))
                    if generated_logprob is not None and generated_token_count > 0
                    else 1e6
                ),
                "generated_token_count": generated_token_count,
                "ground_truth_logprob": ground_truth_logprob,
                "ground_truth_perplexity": (
                    np.exp(-ground_truth_logprob / len(gt_ids))
                    if ground_truth_logprob is not None and len(gt_ids) > 0
                    else None
                ) if ground_truth else None,
            }


        if TASK != Tasks.OQA:
            # Clean result - remove any assistant tags or extra text
            predicted_text = predicted_text.replace('[assistant]', '').strip()
            if '[system]' in predicted_text:
                predicted_text = predicted_text.split('[system]')[0].strip()

            predicted_text = predicted_text.split('\n')[0].split(';')[0].strip()  # Take first part
            
        if logging:
            print(f"Predicted: {predicted_text}")
            print(f"Probability Slice {probabilities}")
            print(f"Actual: {prompt_dict['ground_truth']}")
            print("-" * 50)

        return predicted_text, probabilities

    except Exception as e:
        print(f"Model generation failed: {e}")
        return None


dataset = DataLoader(Dataset_t.JSON)
df_sampled = dataset.data

prompts = []

for i, sample in df_sampled.iterrows():
    if dataset.dataset_t == Dataset_t.CSV:
        assert TASK != Tasks.QA, "You are loading CSV but code is assuming JSON"
        diagnosis = sample["ICD9 Diagnosis"]
        note = sample["Note"]

        prompts.append({
            "prompt_id": f"classify_{i}",
            "prompt": note,
            "ground_truth": diagnosis,
        })
    elif dataset.dataset_t == Dataset_t.JSON:
        prompts.append({
            'prompt_id': f"qa_{i}",
            'prompt': sample["context"],
            'question': sample['question'],
            'ground_truth': sample['answer_text']
        })
        

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

    for i in tqdm(range(len(prompts) - offset)):
        prompt = prompts[i]
        obj = {}
        inference, probs = run_inference(prompt, dataset.code_to_title)
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