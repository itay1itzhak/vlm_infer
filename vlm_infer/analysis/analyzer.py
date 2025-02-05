from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .metrics import compute_metrics, compute_accuracy

class ResultAnalyzer:
    """Analyze model results with various groupings and metrics."""
    
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df
        
    def add_scores(self, model_column: str) -> pd.DataFrame:
        """Add accuracy scores for a model's predictions."""
        correctness_column = f"{model_column}_correctness"
        processed_column = f"{model_column}_processed"
        
        # Calculate final score for each sample
        final_scores = []
        for _, row in self.df.iterrows():
            if correctness_column in self.df.columns:
                label = row[correctness_column]
                if label == "CORRECT":
                    score = 1.0
                elif label == "PARTIALLY_CORRECT":
                    score = 0.5
                else:  # Fall back to string matching
                    score = compute_accuracy(
                        row[processed_column],
                        row["expected_answer"]
                    )
            else:
                # Use string matching if no correctness labels
                score = compute_accuracy(
                    row[processed_column],
                    row["expected_answer"]
                )
            final_scores.append(score)
        
        # Add final scores column
        self.df[f"{model_column}_score"] = final_scores
        return self.df
    
    def analyze_by_group(
        self,
        group_column: str,
        model_column: str
    ) -> pd.DataFrame:
        """Analyze results grouped by a specific column."""
        return self.df.groupby(group_column).agg({
            f"{model_column}_score": ["mean", "std", "count"]
        }).round(3)
    
    def analyze_by_story(
        self,
        model_column: str
    ) -> pd.DataFrame:
        """Analyze results grouped by story."""
        return self.analyze_by_group("story_structure", model_column)
    
    def analyze_by_feature(
        self,
        feature_column: str,
        model_column: str
    ) -> pd.DataFrame:
        """Analyze results grouped by any feature."""
        return self.analyze_by_group(feature_column, model_column)
    
    def compare_models(
        self,
        model_columns: List[str],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """Compare multiple models' performance."""
        metrics = {}
        
        if group_by:
            for group in self.df[group_by].unique():
                group_df = self.df[self.df[group_by] == group]
                metrics[group] = {
                    model: {
                        "accuracy": group_df[f"{model}_score"].mean(),
                        "accuracy_std": group_df[f"{model}_score"].std()
                    }
                    for model in model_columns
                }
        else:
            metrics["overall"] = {
                model: {
                    "accuracy": self.df[f"{model}_score"].mean(),
                    "accuracy_std": self.df[f"{model}_score"].std()
                }
                for model in model_columns
            }
        
        return pd.DataFrame(metrics).round(3) 