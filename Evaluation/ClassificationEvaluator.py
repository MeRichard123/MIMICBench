from Generic.Evaluator import Evaluator
from sklearn.metrics import (
    f1_score, recall_score, accuracy_score, precision_score, 
    balanced_accuracy_score, confusion_matrix
)
from bert_score import score as bert_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re


class ClassificationEvaluator(Evaluator):
    def __init__(self, model, use_as_judge):
        super().__init__(model, use_as_judge)
        self.use_as_judge = use_as_judge
        self.model = model
    
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


    def evaluate(self, ground_truth, predictions):
        accuracy = accuracy_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions, average='weighted', zero_division=1)
        recall = recall_score(ground_truth, predictions, average='weighted', zero_division=1)
        f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=1)
        balanced_accuracy = balanced_accuracy_score(ground_truth, predictions)

        classes = sorted(set(ground_truth))
        cm = confusion_matrix(ground_truth, predictions, labels=classes)

        P, R, F1 = bert_score(list(predictions), list(ground_truth), lang="en", rescale_with_baseline=True)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
        if self.use_as_judge:
            scores = []
            for idx in tqdm(range(len(list(zip(predictions, ground_truth))))):
                pred, ground = list(zip(predictions, ground_truth))[idx]
                if pred == "N/A":
                    continue
                if len(pred) < 3:
                    scores.append(0)
                    continue
                llm_as_judge = self.LLM_as_Judge(pred, ground, self.build_eval_prompt)
                match = re.search(r"Score\s*:\s*(\d+)", llm_as_judge)
                if match:
                    score = int(match.group(1))
                    scores.append(score)
                else:
                    scores.append(0)

            print(f"Average LLM Judge Score: {np.mean(scores):.4f}")

        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Diagnosis')
        plt.ylabel('Ground Truth Diagnosis')
        plt.title('Confusion Matrix')
        plt.savefig(f"confmatrix-{self.model}.png")


if __name__ == "__main__":
    e = ClassificationEvaluator()