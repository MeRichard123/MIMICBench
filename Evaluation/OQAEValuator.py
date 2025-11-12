from Generic.Evaluator import Evaluator

class OQAEvaluator(Evaluator):
    def __init__(self, model, use_as_judge):
        super().__init__(model, use_as_judge)
        self.use_as_judge = use_as_judge
        self.model = model

    def evaluate(self, ground_truth, predictions):
        return super().evaluate(ground_truth, predictions)