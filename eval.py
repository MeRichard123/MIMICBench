import os, json
import numpy as np
from Evaluation.ClassificationEvaluator import ClassificationEvaluator


BASE_DIR = "/workspaces/CMP9794-Advanced-Artificial-Intelligence/Results/"
MODEL = "Qwen-2.5-7b-Medical"

# Eval as Classification

with open(os.path.join(BASE_DIR, MODEL + ".json")) as fp:
    parse = json.load(fp)

    predictions = np.array([data['predicted_diagnosis'] for data in parse])

    ground_truth = np.array([data['ground_truth'].strip() for data in parse])
    predicted = [p if p else "N/A" for p in predictions]

    evaluator = ClassificationEvaluator(MODEL, False)
    evaluator.evaluate(ground_truth, predicted)


# Eval as QA