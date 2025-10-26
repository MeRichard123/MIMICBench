import os, re, json
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, balanced_accuracy_score

BASE_DIR = "/workspaces/CMP9794-Advanced-Artificial-Intelligence/"


def extract_icd9_code(predicted_response):
    icd9_pattern = r'\b[a-zA-Z]?\d+\b'

    # Find all matching ICD-9 codes
    matches = re.findall(icd9_pattern, predicted_response)

    # Return the first match if any exist
    return matches[0] if matches else ''


with open(os.path.join(BASE_DIR, "Llama-3.1-8B-Base.json")) as fp:
    parse = json.load(fp)

    predictions = np.array([extract_icd9_code(data['predicted_diagnosis']) for data in parse])

    ground_truth = np.array([data['ground_truth'] for data in parse])

    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='weighted', zero_division=1)
    recall = recall_score(ground_truth, predictions, average='weighted', zero_division=1)
    f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=1)
    balanced_accuracy = balanced_accuracy_score(ground_truth, predictions)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

