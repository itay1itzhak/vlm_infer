import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from vlm_infer.analysis import (
    AnswerEvaluator, ResultAnalyzer, ResultPlotter,
    compute_metrics, compute_f1_score
)
from .utils import create_test_results

@pytest.fixture
def test_results():
    """Create test results dataframe."""
    return create_test_results()

def test_compute_f1_score():
    """Test F1 score computation."""
    assert compute_f1_score("the cat", "the cat") == 1.0
    assert compute_f1_score("the cat", "a dog") == 0.0
    assert compute_f1_score("the black cat", "the cat") > 0.0

def test_compute_metrics():
    """Test metrics computation."""
    predictions = ["the cat", "a dog", "blue sky"]
    targets = ["the cat", "the dog", "cloudy sky"]
    
    metrics = compute_metrics(predictions, targets)
    assert "f1" in metrics
    assert "f1_std" in metrics
    assert "exact_match" in metrics
    assert 0 <= metrics["f1"] <= 1
    assert metrics["exact_match"] > 0

def test_answer_evaluator():
    """Test answer evaluator."""
    evaluator = AnswerEvaluator(extraction_method="regex")
    df = pd.DataFrame({
        "model_generation_test": [
            "I think the answer is yes.",
            "No, because...",
            "Therefore, maybe."
        ]
    })
    
    processed_df = evaluator.process_answers(df, "model_generation_test")
    assert "model_generation_test_processed" in processed_df.columns

def test_result_analyzer(test_results):
    """Test result analyzer."""
    analyzer = ResultAnalyzer(test_results)
    
    # Test adding scores
    df = analyzer.add_scores("model_generation_llava")
    assert "model_generation_llava_f1" in df.columns
    
    # Test group analysis
    group_analysis = analyzer.analyze_by_group("story_type", "model_generation_llava")
    assert len(group_analysis) == 2  # type_A and type_B
    
    # Test model comparison
    comparison = analyzer.compare_models(
        ["model_generation_llava", "model_generation_gpt4v"]
    )
    assert "overall" in comparison.columns

def test_result_plotter(test_results, tmp_path):
    """Test result plotter."""
    plotter = ResultPlotter(save_dir=str(tmp_path))
    
    # Test model comparison plot
    plotter.plot_model_comparison(
        test_results,
        save_name="test_comparison"
    )
    assert (tmp_path / "test_comparison.png").exists()
    
    # Test feature analysis plot
    plotter.plot_feature_analysis(
        test_results,
        "story_type",
        "model_generation_llava",
        save_name="test_feature"
    )
    assert (tmp_path / "test_feature.png").exists() 