import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from rich.logging import RichHandler

from vlm_infer.config import Config
from vlm_infer.data import get_dataset
from vlm_infer.models import get_model
from vlm_infer.utils.logger import setup_logger
from vlm_infer.inference import InferencePipeline
from vlm_infer.types import InferenceType
from vlm_infer.utils.env import load_env_vars

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VLM Inference Pipeline")
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config YAML file")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Name of the model to use (e.g., llava, gpt4v)")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Name of the dataset to use (e.g., coco, vtom)")
    
    # Optional arguments
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Optional path to dataset directory (overrides config data_dir)")
    parser.add_argument("--output_dir", type=str, default="vlm_infer/results",
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode (only process first 2 examples)")
 
    # Add inference type argument
    parser.add_argument("--inference_type", type=str, default="standard",
                       choices=[t.value for t in InferenceType],
                       help="Type of inference to perform. Options are: standard (image + question), image_story (image + story + question), story_only (story + question), image_captions (image + captions + question), captions_only (captions + question), story_captions (story + captions + question)")
    
    # Add environment file argument
    parser.add_argument("--env_file", type=str, default=None,
                       help="Path to .env file")
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Parse arguments
    args = parse_args()
    
    # Load environment variables
    load_env_vars(args.env_file)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(
        name="vlm_infer",
        log_file=output_dir / "inference.log",
        log_level=log_level
    )
    
    # Load configuration
    config_dict = load_config(args.config)
    
    # Ensure dataset configuration exists
    if "dataset" not in config_dict:
        config_dict["dataset"] = {}
    
    # Override dataset path if provided via command line
    if args.dataset_path:
        config_dict["dataset"]["data_dir"] = args.dataset_path
    
    # Create config object with validation
    config = Config(config_dict)
    
    logger.info(f"Starting inference with model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset}")
    if args.dataset_path:
        logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Device: {args.device}")
    

    
    try:
        # Initialize model
        model = get_model(
            model_name=args.model_name,
            config=config.model_config,
            device=args.device,
        )
        
        # Initialize dataset
        dataset = get_dataset(
            dataset_name=args.dataset,
            config=config.dataset_config
        )
        
        # If in debug mode, limit dataset size
        if args.debug:
            logger.info("Running in debug mode with first 2 examples")
            dataset.data = dataset.data.iloc[:2]
        
        # Initialize inference pipeline
        pipeline = InferencePipeline(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            output_dir=output_dir,
            config=config
        )
        
        # Run inference
        results = pipeline.run()
        
        # Save results
        pipeline.save_results(results)
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise
    
if __name__ == "__main__":
    main()
