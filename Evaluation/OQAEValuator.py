from Generic.Evaluator import Evaluator
from bert_score import score as bert_score
import evaluate
from nltk.tokenize import word_tokenize
from rouge import Rouge 
import nltk
import numpy as np

nltk.download('punkt_tab')

import torch

class OQAEvaluator(Evaluator):
    def __init__(self, model, use_as_judge):
        super().__init__(model, use_as_judge)
        self.use_as_judge = use_as_judge
        self.model = model


    def evaluate(self, ground_truth, predictions, probs):
        P, R, F1 = bert_score(list(predictions), list(ground_truth), lang='en', rescale_with_baseline=True)

        P = torch.tensor(P) if isinstance(P, (tuple, str)) else P
        R = torch.tensor(R) if isinstance(R, (tuple, str)) else R
        F1 = torch.tensor(F1) if isinstance(F1, (tuple, str)) else F1

        bleu = evaluate.load("bleu")
        results = bleu.compute(predictions=predictions, references=ground_truth, tokenizer=word_tokenize)

        kl_scores = [
            self.calculate_kl_divergence(prob_dict, actual)
            for prob_dict, actual in zip(probs, ground_truth)
        ]
        kl_avg = np.mean(kl_scores)

        perplexity = np.array([prob['ground_truth_perplexity'] for prob in probs])


        print(f"BERTScore F1: {F1.mean():.4f} ± {np.nanstd(F1):.4f}")
        print(f"BERTScore Recall: {R.mean():.4f} ± {np.nanstd(R):.4f}")
        print(f"BERTScore Precision: {P.mean():.4f} ± {np.nanstd(P):.4f}")

        print(f"KL-Divergence: {kl_avg:.4f}")
        print(f"Perplexity: {np.mean(perplexity):.2f} ± {np.std(perplexity):.2f}")

        print(f"BLEU {results['bleu']:.4f}")
        