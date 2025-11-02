import pandas as pd
from abc import ABC, abstractmethod
import os
from unsloth import FastLanguageModel
import torch
import re

class Evaluator(ABC):
    BASE_DIR = "/workspaces/CMP9794-Advanced-Artificial-Intelligence/"
    def __init__(self, model, use_as_judge):
        self.DATA_MODEL = model
        self.__pairs = Evaluator.get_pair_dictionary()

        if use_as_judge:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = "openai/gpt-oss-20b",
                dtype = None,
                max_seq_length = 4096,
                load_in_4bit = True,
                device_map="auto",  # Let HF automatically distribute between GPU/CPU
                offload_folder=os.path.join(os.getcwd(), "offload"),
                full_finetuning=False,
            )

            FastLanguageModel.for_inference(self.model)

    @classmethod
    def get_pair_dictionary(cls):
        df = pd.read_csv(os.path.join(cls.BASE_DIR, "MIMIC-SAMPLED.csv"))
        df_copy = df.copy()
        df_copy.dropna(subset=["ICD9 Diagnosis"], inplace=True)
        df_copy.dropna(subset=["note_category"], inplace=True)
        unique_pairs = df_copy[['ICD9 Diagnosis', 'SHORT_TITLE']].drop_duplicates()
        return dict(unique_pairs.values)
    
    def LLM_as_Judge(self, y_pred, ground, prompt_builder):
        """Run inference using the pre-loaded model"""
        # Get device from model (don't reassign device)
        device = next(self.model.parameters()).device
        prompt = prompt_builder(y_pred, ground, self.__pairs)
        # Tokenize
        inputs = self.tokenizer(
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
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256, 
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=0.1,
                    repetition_penalty=1.0,
                )

            # Decode only new tokens
            generated_tokens = outputs[0][input_ids.shape[1]:]
            result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            return result

        except Exception as e:
            print(f"Model generation failed: {e}")
            return None
    
    @classmethod
    def extract_icd9_code(cls, result):
        if "Code:" in result:
            result = result.split("Code:")[-1]
        match = re.search(r"\b[A-Z]\d{1,3}\.?[A-Z0-9]*\b", result)
        return match.group(0).strip() if match else result.strip()
    
    @abstractmethod
    def evaluate(self, ground_truth, predictions):
        return

    


