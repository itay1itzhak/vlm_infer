import argparse
from pathlib import Path
import pandas as pd
import yaml
import logging
from typing import List, Dict, Any, Optional
import glob

from vlm_infer.utils.logger import setup_logger
from vlm_infer.analysis import AnswerEvaluator, ResultAnalyzer, ResultPlotter

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis on VLM inference results")
    
    # Results directory
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing results files")
    
    # Analysis options
    parser.add_argument("--models", type=str, nargs="+", default=None,
                       help="Specific models to analyze. Default: all available")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                       help="Specific datasets to analyze. Default: all available")
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
    parser.add_argument("--output_dir", type=str, default="analysis_results",
                       help="Directory to save analysis results")
    parser.add_argument("--save_plots", action="store_true",
                       help="Save plots instead of displaying them")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()

def find_results_files(results_dir: Path, dataset: Optional[str] = None) -> List[Path]:
    """Find all results files in directory."""
    pattern = f"*{dataset if dataset else ''}*.csv" if dataset else "*.csv"
    return list(results_dir.glob(pattern))

def load_results(file_path: Path) -> pd.DataFrame:
    """Load results from a CSV file."""
    return pd.read_csv(file_path)

def detect_models(df: pd.DataFrame) -> List[str]:
    """Detect model columns in the dataframe."""
    return [col for col in df.columns if col.startswith("model_generation_")]

def run_analysis(args: argparse.Namespace):
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        name="vlm_analysis",
        log_file=output_dir / "analysis.log",
        log_level=logging.DEBUG if args.verbose else logging.INFO
    )
    
    # Initialize components
    evaluator = AnswerEvaluator(extraction_method="regex")
    plotter = ResultPlotter(save_dir=output_dir if args.save_plots else None)
    
    # Find and load results
    results_dir = Path(args.results_dir)
    results_files = []
    
    if args.datasets:
        for dataset in args.datasets:
            results_files.extend(find_results_files(results_dir, dataset))
    else:
        results_files = find_results_files(results_dir)
    
    if not results_files:
        logger.error(f"No results files found in {results_dir}")
        return
    
    # Process each results file
    for file_path in results_files:
        logger.info(f"Processing {file_path}")
        df = load_results(file_path)
        
        # Detect available models
        available_models = detect_models(df)
        models_to_analyze = [m for m in (args.models or available_models) 
                           if m in available_models]
        
        if not models_to_analyze:
            logger.warning(f"No matching models found in {file_path}")
            continue
        
        # Process answers for each model
        for model in models_to_analyze:
            df = evaluator.process_answers(df, model)
        
        # Initialize analyzer
        analyzer = ResultAnalyzer(df)
        
        # Add F1 scores
        for model in models_to_analyze:
            analyzer.add_scores(model)
        
        # Run analyses
        dataset_name = file_path.stem
        
        # Model comparison
        if not args.skip_model_comparison and len(models_to_analyze) > 1:
            logger.info("Running model comparison analysis")
            comparison = analyzer.compare_models(models_to_analyze)
            comparison.to_csv(output_dir / f"{dataset_name}_model_comparison.csv")
            
            plotter.plot_model_comparison(
                comparison,
                title=f"Model Comparison - {dataset_name}",
                save_name=f"{dataset_name}_model_comparison"
            )
        
        # Feature analysis
        if not args.skip_feature_analysis:
            logger.info("Running feature analysis")
            for feature in args.features:
                if feature not in df.columns:
                    logger.warning(f"Feature {feature} not found in dataset")
                    continue
                    
                for model in models_to_analyze:
                    analysis = analyzer.analyze_by_feature(feature, model)
                    analysis.to_csv(
                        output_dir / f"{dataset_name}_{model}_{feature}_analysis.csv"
                    )
                    
                    plotter.plot_feature_analysis(
                        df, feature, model,
                        title=f"{model} Performance by {feature}",
                        save_name=f"{dataset_name}_{model}_{feature}_analysis"
                    )
        
        # Story analysis
        if not args.skip_story_analysis:
            logger.info("Running story analysis")
            for model in models_to_analyze:
                analysis = analyzer.analyze_by_story(model)
                analysis.to_csv(
                    output_dir / f"{dataset_name}_{model}_story_analysis.csv"
                )
        
        # Model comparison by feature
        if not args.skip_model_comparison and len(models_to_analyze) > 1:
            for feature in args.features:
                if feature not in df.columns:
                    continue
                    
                plotter.plot_model_comparison_by_feature(
                    df, feature, models_to_analyze,
                    title=f"Model Comparison by {feature}",
                    save_name=f"{dataset_name}_model_comparison_{feature}"
                )
    
    logger.info("Analysis completed!")

def main():
    args = parse_args()
    run_analysis(args)

if __name__ == "__main__":
    main() 