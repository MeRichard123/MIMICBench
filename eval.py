import os, json, sys
import numpy as np
from Evaluation.ClassificationEvaluator import ClassificationEvaluator
from Evaluation.QAEvaluator import QAEvaluator
from Evaluation.OQAEValuator import OQAEvaluator

shift = lambda args: args[1:]

def parse_QA_reponse(text):
    if ')' in text: # contains a repsonse in the form a) 
        remove_option = text.split(")")[1]
        if ':' in remove_option:
            return remove_option.split(":")[0].strip()
        else:
            return remove_option.strip()
    return text.strip()

BASE_DIR = "/workspaces/CMP9794-Advanced-Artificial-Intelligence/Results/"
MODEL = "meditron-7b"

if not (shift(sys.argv)):
    print("""[usage]
    > python eval.py [task]
    
    **tasks**
          -mcqa : Multi Choice Question Answering
          -cls: Classification
          -oqa: Open Question Answering
    """)


match shift(sys.argv)[0]:

    case "-cls":
        # Eval as Classification
        with open(os.path.join(BASE_DIR, MODEL + "-classification.json")) as fp:
            parse = json.load(fp)

            predictions = np.array([data['predicted_diagnosis'] for data in parse])
            ground_truth = np.array([data['ground_truth'].strip() for data in parse])
            probabilities = np.array([data['probabilities'] for data in parse])
            
            predicted = [p if p else "N/A" for p in predictions]

            evaluator = ClassificationEvaluator(MODEL, False)
            evaluator.evaluate(ground_truth, predicted, probabilities)

    case "-mcqa":
        # Eval as QA
        with open(os.path.join(BASE_DIR, MODEL + "-mcqa.json")) as fp:
            parse = json.load(fp)

        predictions = np.array([parse_QA_reponse(data['predicted_diagnosis']) for data in parse])

        ground_truth = np.array([data['ground_truth'].strip() for data in parse])
        predicted = [p if p else "N/A" for p in predictions]
        probabilities = np.array([data['probabilities'] for data in parse])

        evaluator = QAEvaluator(MODEL, False)
        evaluator.evaluate(ground_truth, predicted, probabilities)

    case "-oqa":
        # Eval as Open QA
        with open(os.path.join(BASE_DIR, MODEL + "-oqa.json")) as fp:
            parse = json.load(fp)

        predictions = np.array([data['predicted_diagnosis'] for data in parse])

        ground_truth = np.array([data['ground_truth'].strip() for data in parse])
        predicted = [p if p else "N/A" for p in predictions]

        evaluator = OQAEvaluator(MODEL, False)
        evaluator.evaluate(ground_truth, predicted)