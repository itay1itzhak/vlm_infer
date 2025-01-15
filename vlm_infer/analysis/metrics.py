from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    return answer.lower().strip()

def compute_f1_score(pred: str, target: str) -> float:
    """Compute F1 score between prediction and target."""
    pred_tokens = set(normalize_answer(pred).split())
    target_tokens = set(normalize_answer(target).split())
    
    common = pred_tokens & target_tokens
    if not common:
        return 0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def compute_metrics(
    predictions: List[str],
    targets: List[str]
) -> Dict[str, float]:
    """Compute all metrics for given predictions and targets."""
    f1_scores = [
        compute_f1_score(pred, target)
        for pred, target in zip(predictions, targets)
    ]
    
    return {
        "f1": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "exact_match": np.mean([
            normalize_answer(p) == normalize_answer(t)
            for p, t in zip(predictions, targets)
        ])
    } 