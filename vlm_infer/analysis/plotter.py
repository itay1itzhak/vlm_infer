from typing import List, Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ResultPlotter:
    """Plot analysis results in various formats."""
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_or_show(self, name: Optional[str] = None):
        if name and self.save_dir:
            plt.savefig(self.save_dir / f"{name}.png", bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def plot_model_comparison(
        self,
        results: pd.DataFrame,
        metric: str = "f1",
        title: str = "Model Comparison",
        save_name: Optional[str] = None
    ):
        """Plot comparison of multiple models."""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results, y=metric)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_or_show(save_name)
    
    def plot_feature_analysis(
        self,
        results: pd.DataFrame,
        feature: str,
        model_column: str,
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ):
        """Plot analysis by feature."""
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=self.df,
            x=feature,
            y=f"{model_column}_f1"
        )
        plt.title(title or f"Performance by {feature}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_or_show(save_name)
    
    def plot_model_comparison_by_feature(
        self,
        results: pd.DataFrame,
        feature: str,
        model_columns: List[str],
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ):
        """Plot comparison of multiple models grouped by feature."""
        melted = pd.melt(
            results,
            id_vars=[feature],
            value_vars=[f"{m}_f1" for m in model_columns],
            var_name="Model",
            value_name="F1 Score"
        )
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=melted,
            x=feature,
            y="F1 Score",
            hue="Model"
        )
        plt.title(title or f"Model Comparison by {feature}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_or_show(save_name) 