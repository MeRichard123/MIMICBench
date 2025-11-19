from Generic.Evaluator import Evaluator
from bert_score import score as bert_score
import evaluate
from nltk.tokenize import word_tokenize
from rouge import Rouge 
import nltk

nltk.download('punkt_tab')

import torch

class OQAEvaluator(Evaluator):
    def __init__(self, model, use_as_judge):
        super().__init__(model, use_as_judge)
        self.use_as_judge = use_as_judge
        self.model = model

    def evaluate(self, ground_truth, predictions):
        P, R, F1 = bert_score(list(predictions), list(ground_truth), lang='en', rescale_with_baseline=True)

        P = torch.tensor(P) if isinstance(P, (tuple, str)) else P
        R = torch.tensor(R) if isinstance(R, (tuple, str)) else R
        F1 = torch.tensor(F1) if isinstance(F1, (tuple, str)) else F1

        rouge = Rouge()
        R_F, R_P, R_R = rouge.get_scores(list(predictions), list(ground_truth), avg=True)['rouge-l']

        bleu = evaluate.load("bleu")
        results = bleu.compute(predictions=predictions, references=ground_truth, tokenizer=word_tokenize)


        print(f"BERTScore F1: {F1.mean():.4f}")
        print(f"BERTScore Recall: {R.mean():.4f}")
        print(f"BERTScore Precision: {P.mean():.4f}")
        print(f"ROUGE F1: {R_F}")
        print(f"ROUGE Precision: {R_P}")
        print(f"ROUGE Recall: {R_R}")
        print(f"BLEU {results['bleu']:.4f}")
        