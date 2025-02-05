from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score
import re
def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    return answer.lower().strip()

def rule_based_accuracy(pred: str, target: str) -> float:
    """Rule-based accuracy for correctness labels."""
    pred, target = pred.lower().strip(), target.lower().strip()
    # Remove from pred unncessary words like "the", "a", "an", "it", "will", "be", etc.
    pred = re.sub(r'\bthe\b|\ba\b|\ban\b|\bit\b|\bwill\b|\bbe\b|\bis\b|\bare\b|\bwas\b|\bwere\b', '', pred)
    # Remove expressions like "In the ", "In a ", "In an ", "In the end", "In the beginning", etc.
    pred = re.sub(r'\bIn the\b|\bIn a\b|\bIn an\b|\bIn the end\b|\bIn the beginning\b', '', pred)
    # Remove from target unncessary words like "the", "a", "an", etc.
    target = re.sub(r'\bthe\b|\ba\b|\ban\b', '', target)
    exact_match = normalize_answer(pred) == normalize_answer(target)
    if exact_match:
        return 1.0
    # Check if target is a substring of pred
    if target in pred:
        return 0.5
    return 0.0

def compute_accuracy(pred: str, target: str) -> float:
    """Compute accuracy between prediction and target."""
    if pred == 'NO_ANSWER' or pred == 'INCORRECT' or pred == 'NOT_CLEAR':
        return 0.0
    if normalize_answer(pred) == normalize_answer(target):
        return 1.0
    if rule_based_accuracy(pred, target):
        return 1.0
    return 0.0

def compute_metrics(
    predictions: List[str],
    targets: List[str],
    correctness_labels: List[str] = None
) -> Dict[str, float]:
    """Compute metrics for predictions."""
    accuracy_scores = []
    
    if correctness_labels:
        for pred, target, label in zip(predictions, targets, correctness_labels):
            if label == "CORRECT":
                accuracy_scores.append(1.0)
            elif label == "PARTIALLY_CORRECT":
                accuracy_scores.append(0.5)
            else:  # INCORRECT, NO_ANSWER, NOT_CLEAR - fall back to string matching
                accuracy_scores.append(compute_accuracy(pred, target))
    else:
        # Use string matching only
        accuracy_scores = [
            compute_accuracy(pred, target)
            for pred, target in zip(predictions, targets)
        ]
    
    return {
        "accuracy": np.mean(accuracy_scores),
        "accuracy_std": np.std(accuracy_scores)
    } 