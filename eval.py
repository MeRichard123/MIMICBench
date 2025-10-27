import os, re, json
import numpy as np
from sklearn.metrics import (
    f1_score, recall_score, accuracy_score, precision_score, 
    balanced_accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from utils import evaluate_model_performance

matplotlib.use('Agg')

BASE_DIR = "/workspaces/CMP9794-Advanced-Artificial-Intelligence/"
MODEL = "MedPhi"

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

    for pred, ground in zip(predicted, ground_truth):
        llm_as_judge = evaluate_model_performance(pred, ground)
        print(llm_as_judge)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Diagnosis')
    plt.ylabel('Ground Truth Diagnosis')
    plt.title('Confusion Matrix')
    plt.savefig(f"confmatrix-{MODEL}.png")

