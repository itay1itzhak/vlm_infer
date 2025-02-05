import argparse
from pathlib import Path
import random
import pandas as pd
import yaml
import logging
from typing import List, Dict, Any, Optional
import glob
from datetime import datetime
import numpy as np

from vlm_infer.utils.logger import setup_logger
from vlm_infer.analysis import AnswerEvaluator, ResultAnalyzer, ResultPlotter
from vlm_infer.analysis.metrics import compute_accuracy

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis on VLM inference results")
    
    # Results directory
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing results files")
    
    # Analysis options
    parser.add_argument("--models", type=str, nargs="+", default=None,
                       help="Specific models to analyze.")
    parser.add_argument("--features", type=str, nargs="+", 
                       default=["story_type", "num_people", "story_structure"],
                       help="Features to analyze")
    
    # Analysis types
    parser.add_argument("--skip_model_comparison", action="store_true",
                       help="Skip model comparison analysis")
    parser.add_argument("--skip_feature_analysis", action="store_true",
                       help="Skip feature-based analysis")
    parser.add_argument("--skip_story_analysis", action="store_true",
                       help="Skip story-based analysis")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="vlm_infer/results/analysis_results",
                       help="Directory to save analysis results")
    parser.add_argument("--save_plots", action="store_true",
                       help="Save plots instead of displaying them")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    # Add new arguments for extraction
    parser.add_argument(
        "--extraction_method",
        type=str,
        choices=["regex", "lm"],
        default="regex",
        help="Method to extract answers from model outputs"
    )
    parser.add_argument(
        "--extraction_model",
        type=str,
        default="allenai/OLMo-2-1124-7B-Instruct",
        help="Model to use for extraction"
    )
    parser.add_argument(
        "--extraction_type",
        type=str,
        choices=["base", "with_question", "with_answer", "base_with_examples"],
        default="base",
        help="Type of extraction prompt to use"
    )
    
    # Add correctness method argument
    parser.add_argument(
        "--correctness_method",
        type=str,
        choices=["regex", "lm"],
        default=None,
        help="Method to check answer correctness. If None, uses same as extraction method"
    )
    
    # Add correctness type argument
    parser.add_argument(
        "--correctness_type",
        type=str,
        choices=["base", "with_question", "with_answer", "base_with_examples", "soft", "soft_with_question", "soft_with_question_with_examples", "soft_with_question_with_answer_with_examples"],
        default="base",
        help="Type of correctness check prompt to use"
    )
    
    # Add examples option
    parser.add_argument(
        "--use_examples",
        action="store_true",
        help="Include examples in prompts"
    )
    
    return parser.parse_args()

def get_dataset_name(results_dir: Path) -> str:
    """Get dataset name from results directory path."""
    # The dataset name is the parent directory name
    return results_dir.parent.name

def find_results_files(results_dir: Path) -> Dict[str, Path]:
    """
    Find all results files in directory.
    Structure is results/model_name/run_timestamp/results.csv
    Returns a dict mapping model names to their latest results file
    """
    model_results = {}
    
    # Check if dir is a specific run dir
    does_have_subdirs = any(p.is_dir() for p in results_dir.iterdir())
    if results_dir.is_dir() and not does_have_subdirs and any(results_dir.glob("*.csv")):
        # Use parent dir name as model name
        model_name = results_dir.parent.parent.name
        model_results[model_name] = results_dir / "results.csv"
        return model_results
    
    # Otherwise look for model directories
    for model_dir in results_dir.glob("*"):
        if model_dir.is_dir():
            model_name = model_dir.name
            # Get the latest run
            run_dirs = sorted(model_dir.glob("*"), key=lambda x: x.name)
            if run_dirs:
                latest_run = run_dirs[-1]
                csv_files = list(latest_run.glob("*.csv"))
                if csv_files:
                    model_results[model_name] = csv_files[0]
    
    return model_results

def load_results(file_path: Path) -> pd.DataFrame:
    """Load results from a CSV file."""
    return pd.read_csv(file_path)

def detect_models(df: pd.DataFrame) -> List[str]:
    """Detect model columns in the dataframe."""
    return [col for col in df.columns if col.startswith("model_generation_")]

def save_analysis_config(
    output_dir: Path,
    results_dir: Path,
    models: List[str],
    features: List[str],
    extraction_model: str,
    extraction_method: str,
    extraction_type: str,
    correctness_type: str
) -> Path:
    """Save analysis configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{extraction_model.split('/')[-1]}_{extraction_method}_{extraction_type}_{correctness_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        "timestamp": timestamp,
        "results_dir": str(results_dir),
        "dataset": get_dataset_name(results_dir),
        "models": models,
        "features": features,
        "extraction_model": extraction_model,
        "extraction_method": extraction_method,
        "extraction_type": extraction_type,
        "correctness_type": correctness_type
    }
    
    with open(run_dir / "analysis_config.yaml", "w") as f:
        yaml.dump(config, f)
        
    return run_dir

def preprocess_features_names(df: pd.DataFrame, features: List[str]) -> List[str]:
    """Preprocess feature names."""
    return [feature if feature in df.columns else "param="+feature for feature in features]

def save_and_log_results(
    df: pd.DataFrame,
    model_name: str,
    analyses_results: Dict[str, pd.DataFrame],
    logger: logging.Logger,
    output_dir: Path,
    dataset_name: str
):
    """Log and save all results and metrics."""
    # Log example results
    row_index = random.randint(0, len(df)-1)
    row = df.iloc[row_index]
    logger.info(f"\nExample results for {model_name} row number {row_index}:")
    logger.info(f"Original answer: {row[f'model_generation']}")
    logger.info(f"Processed answer: {row[f'model_generation_processed']}")
    logger.info(f"Expected answer: {row['expected_answer']}")
    logger.info(f"Correctness: {row[f'model_generation_correctness']}")
    logger.info(f"Final score: {row[f'model_generation_score']:.3f}")

    # Calculate and log correctness distribution
    correctness_counts = df[f"model_generation_correctness"].value_counts()
    correctness_percentages = df[f"model_generation_correctness"].value_counts(normalize=True) * 100

    logger.info("\nCorrectness Distribution:")
    for label in ["CORRECT", "PARTIALLY_CORRECT", "INCORRECT", "NO_ANSWER", "NOT_CLEAR"]:
        if label in correctness_counts:
            logger.info(f"{label}: {correctness_counts[label]} ({correctness_percentages[label]:.2f}%)")

    # Calculate overall accuracy from scores
    scores = df[f"model_generation_score"]
    accuracy = scores.mean()
    accuracy_std = scores.std()

    logger.info(f"\nAccuracy Metrics:")
    logger.info(f"\nOverall Accuracy: {accuracy:.3f}\n")

    # Calculate accuracy by correctness type
    logger.info("\nAccuracy by Correctness Type:")
    for label in ["CORRECT", "PARTIALLY_CORRECT", "INCORRECT", "NO_ANSWER", "NOT_CLEAR"]:
        mask = df[f"model_generation_correctness"] == label
        if mask.any():
            label_scores = df[mask][f"model_generation_score"]
            if not label_scores.empty:
                label_acc = label_scores.mean()
                label_std = label_scores.std()
                logger.info(f"{label}: {label_acc:.3f} ± {label_std:.3f} (n={len(label_scores)})")

    # Log analyses results
    if analyses_results:
        logger.info("\nAnalyses Results:")
        for analysis_name, analysis_df in analyses_results.items():
            logger.info(f"\n{analysis_name.replace('_', ' ').title()}:")
            logger.info(analysis_df)

    # Save final results
    columns_to_move = [
        f"model_generation",
        f"model_generation_processed",
        f"model_generation_correctness",
        f"model_generation_score",
        "expected_answer"
    ]
    other_columns = [col for col in df.columns if col not in columns_to_move]
    df = df[other_columns + columns_to_move]
    df.to_csv(output_dir / f"{dataset_name}_{model_name}_results_with_scores.csv")

def analyze_single_model(
    model_name: str,
    results_file: Path,
    features: List[str],
    evaluator: AnswerEvaluator,
    plotter: ResultPlotter,
    dataset_name: str,
    output_dir: Path,
    logger: logging.Logger,
    skip_feature_analysis: bool = False,
    skip_story_analysis: bool = False
) -> pd.DataFrame:
    """Analyze results for a single model."""
    logger.info(f"Processing {model_name} results from {results_file}", extra={"no_timestamp": True})
    
    # 1. Load and process results
    df = load_results(results_file)
    df = evaluator.process_answers(df, f"model_generation")
    df = evaluator.process_correctness(df, f"model_generation")

    # Log bad format count if any
    if evaluator.get_bad_format_count() > 0:
        logger.info(f"Bad format count: {evaluator.get_bad_format_count()}")
    
    # 2. Calculate scores
    analyzer = ResultAnalyzer(df)
    df = analyzer.add_scores(f"model_generation")
    
    # 3. Run analyses
    analyses_results = {}
    
    # Feature analysis
    if not skip_feature_analysis:
        logger.info(f"Running feature analysis for {model_name}")
        features = preprocess_features_names(df, features)
        for feature in features:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in dataset")
                continue
                
            analysis = analyzer.analyze_by_feature(feature, f"model_generation")
            analyses_results[f"by_{feature}"] = analysis
            
            # Save feature analysis
            output_dir.mkdir(parents=True, exist_ok=True)
            analysis.to_csv(output_dir / f"{dataset_name}_{model_name}_{feature}_analysis.csv")
            
            # Plot feature analysis
            plotter.plot_feature_analysis(
                df, feature, f"model_generation",
                title=f"{model_name} Performance by {feature}",
                save_name=f"{dataset_name}_{model_name}_{feature}_analysis"
            )
    
    # Story analysis
    if not skip_story_analysis:
        logger.info(f"Running story analysis for {model_name}")
        analysis = analyzer.analyze_by_story(f"model_generation")
        analyses_results["by_story"] = analysis
        analysis.to_csv(output_dir / f"{dataset_name}_{model_name}_story_analysis.csv")
    
    # 4. Log and save final results
    save_and_log_results(
        df=df,
        model_name=model_name,
        analyses_results=analyses_results,
        logger=logger,
        output_dir=output_dir,
        dataset_name=dataset_name
    )
    
    return df

def run_model_comparison(
    all_results: Dict[str, pd.DataFrame],
    features: List[str],
    plotter: ResultPlotter,
    dataset_name: str,
    output_dir: Path,
    logger: logging.Logger
):
    """Run comparison analysis across models."""
    if len(all_results) < 2:
        logger.info("Not enough models for comparison analysis")
        return
        
    logger.info("Running model comparison analysis")
    
    # Combine results
    combined_df = pd.concat(all_results.values(), keys=all_results.keys())
    analyzer = ResultAnalyzer(combined_df)
    
    # Overall comparison
    model_columns = [f"model_generation" for m in all_results.keys()]
    comparison = analyzer.compare_models(model_columns)
    
    # Log comparison results
    logger.info("\nModel Comparison Results:")
    logger.info(comparison)
    
    # Save comparison results
    comparison.to_csv(output_dir / f"{dataset_name}_model_comparison.csv")
    
    # Plot overall comparison
    plotter.plot_model_comparison(
        comparison,
        title=f"Model Comparison - {dataset_name}",
        save_name=f"{dataset_name}_model_comparison"
    )
    
    # Comparison by feature
    features = preprocess_features_names(combined_df, features)
    for feature in features:
        if feature not in combined_df.columns:
            continue
            
        # Get feature-specific comparison
        feature_comparison = analyzer.compare_models(model_columns, group_by=feature)
        
        # Log feature comparison
        logger.info(f"\nModel Comparison by {feature}:")
        logger.info(feature_comparison)
        
        # Save feature comparison
        feature_comparison.to_csv(
            output_dir / f"{dataset_name}_model_comparison_{feature}.csv"
        )
        
        # Plot feature comparison
        plotter.plot_model_comparison_by_feature(
            combined_df, 
            feature, 
            model_columns,
            title=f"Model Comparison by {feature}",
            save_name=f"{dataset_name}_model_comparison_{feature}"
        )

def run_analysis(args: argparse.Namespace):
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path(args.results_dir)
    dataset_name = get_dataset_name(results_dir)
    
    run_dir = save_analysis_config(
        output_dir=output_dir,
        results_dir=results_dir,
        models=args.models,
        features=args.features,
        extraction_model=args.extraction_model,
        extraction_method=args.extraction_method,
        extraction_type=args.extraction_type,
        correctness_type=args.correctness_type
    )
    
    logger = setup_logger(
        name="vlm_analysis",
        log_file=run_dir / "analysis.log",
        log_level=logging.DEBUG if args.verbose else logging.INFO
    )
    
    evaluator = AnswerEvaluator(
        extraction_method=args.extraction_method,
        extraction_type=args.extraction_type,
        extraction_model=args.extraction_model,
        correctness_method=args.correctness_method,
        correctness_type=args.correctness_type,
        use_examples=args.use_examples
    )
    plotter = ResultPlotter(save_dir=run_dir if args.save_plots else None)
    
    # Find results files for each model
    model_results = find_results_files(results_dir)
    
    if not model_results:
        logger.error(f"No results files found in {results_dir}")
        return
    
    # Filter models if specified
    if args.models:
        model_results = {k: v for k, v in model_results.items() if k in args.models}
    
    # Analyze each model separately
    all_results = {}
    for model_name, results_file in model_results.items():
        df = analyze_single_model(
            model_name=model_name,
            results_file=results_file,
            features=args.features,
            evaluator=evaluator,
            plotter=plotter,
            dataset_name=dataset_name,
            output_dir=run_dir / model_name,
            logger=logger,
            skip_feature_analysis=args.skip_feature_analysis,
            skip_story_analysis=args.skip_story_analysis
        )
        all_results[model_name] = df
    
    # Run model comparison if needed
    if not args.skip_model_comparison and len(all_results) > 1:
        run_model_comparison(
            all_results=all_results,
            features=args.features,
            plotter=plotter,
            dataset_name=dataset_name,
            output_dir=run_dir,
            logger=logger
        )
    else:
        logger.info("Skipping model comparison analysis")
    
    logger.info("Finished analysis for these models:")
    for model_name in all_results.keys():
        logger.info(f"  - {model_name}")
    logger.info(f"Results saved in {run_dir}")

def main():
    args = parse_args()
    run_analysis(args)

if __name__ == "__main__":
    main() 