from Generic.Evaluator import Evaluator
from bert_score import score as bert_score
import numpy as np
import torch

class QAEvaluator(Evaluator):
    def __init__(self, model, use_as_judge):
        super().__init__(model, use_as_judge)
        self.use_as_judge = use_as_judge
        self.model = model

    def compute_f1_and_balanced_accuracy(self, ground_truth, predictions):
        possible_codes = super().get_pair_dictionary().keys()

        gt_tokens = [set(gt.split()) for gt in ground_truth]
        pred_tokens = [set(pred.split()) for pred in predictions]

        tp = sum(len(gt & pred) for gt, pred in zip(gt_tokens, pred_tokens))
        fp = sum(len(pred) - len(gt & pred) for pred, gt in zip(pred_tokens, gt_tokens)) 
        fn = sum(len(gt) - len(gt & pred) for gt, pred in zip(gt_tokens, pred_tokens)) 
        tn = sum(len(possible_codes - (gt | pred)) for gt, pred in zip(gt_tokens, pred_tokens))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = (sensitivity + specificity) / 2.0

        return f1, balanced_accuracy
    
    def compute_mean_reciprocal_rank(self, ground_truth, predictions):
        ranks = []
        for gt in ground_truth:
            try:
                rank = predictions.index(gt) + 1
                ranks.append(1 / rank)
            except ValueError:
                ranks.append(0)
        return np.mean(ranks)

    def evaluate(self, ground_truth, predictions, probs):
        exact_matches = sum(gt == pred for gt, pred in zip(ground_truth, predictions))
        em = exact_matches / len(ground_truth) if len(ground_truth) > 0 else 0 

        f1, balanced_accuracy = self.compute_f1_and_balanced_accuracy(ground_truth, predictions)

        mrr = self.compute_mean_reciprocal_rank(ground_truth, predictions)

        P, R, F1 = bert_score(list(predictions), list(ground_truth), lang="en", rescale_with_baseline=True)


        kl_scores = [
            self.calculate_kl_divergence(prob_dict, actual)
            for prob_dict, actual in zip(probs, ground_truth)
        ]
        kl_avg = np.mean(kl_scores)

        brier_scores = [
            self.multiclass_brier_score(prob_dict, actual)
            for prob_dict, actual in zip(probs, ground_truth)
        ]
        brier_score = np.mean(brier_scores)


        P = torch.tensor(P) if isinstance(P, (tuple, str)) else P
        R = torch.tensor(R) if isinstance(R, (tuple, str)) else R
        F1 = torch.tensor(F1) if isinstance(F1, (tuple, str)) else F1

        print(f"Exact Match: {em:.4f}")
        print(f"F1 Score {f1:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"Mean Reciprocal Rank: {mrr:.4f}")
        print(f"Brier Score: {brier_score:.4f}")
        print(f"KL-Divergence: {kl_avg:.4f}")
        print(f"BERTScore F1: {F1.mean():.4f} ± {np.std(F1):.4f}")
        print(f"BERTScore Recall: {R.mean():.4f} ± {np.std(R):.4f}")
        print(f"BERTScore Precision: {P.mean():.4f} ± {np.nanstd(P):.4f}")

