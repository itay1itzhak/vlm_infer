from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .metrics import compute_metrics, compute_f1_score

class ResultAnalyzer:
    """Analyze model results with various groupings and metrics."""
    
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df
        
    def add_scores(self, model_column: str) -> pd.DataFrame:
        """Add F1 scores for a model's predictions."""
        self.df[f"{model_column}_f1"] = [
            compute_f1_score(pred, target)
            for pred, target in zip(
                self.df[f"{model_column}_processed"],
                self.df["expected_answer"]
            )
        ]
        return self.df
    
    def analyze_by_group(
        self,
        group_column: str,
        model_column: str
    ) -> pd.DataFrame:
        """Analyze results grouped by a specific column."""
        return self.df.groupby(group_column).agg({
            f"{model_column}_f1": ["mean", "std", "count"]
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
                    model: compute_metrics(
                        group_df[f"{model}_processed"],
                        group_df["expected_answer"]
                    )
                    for model in model_columns
                }
        else:
            metrics["overall"] = {
                model: compute_metrics(
                    self.df[f"{model}_processed"],
                    self.df["expected_answer"]
                )
                for model in model_columns
            }
        
        return pd.DataFrame(metrics).round(3) 