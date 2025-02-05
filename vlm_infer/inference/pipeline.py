from typing import Dict, Any, List, Sequence
import torch
from torch.utils.data import DataLoader
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from PIL import Image
import numpy as np
from datetime import datetime
import yaml

from vlm_infer.models.base import BaseModel
from vlm_infer.data.base import BaseDataset
from vlm_infer.data.vtom import VToMDataset

def custom_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function that handles PIL Images and other data types.
    
    Args:
        batch: Sequence of dictionaries containing batch data
        
    Returns:
        Collated dictionary with properly batched data
    """
    if not batch:
        return {}
    
    collated = {}
    elem = batch[0]
    
    for key in elem:
        try:
            # Special handling for known types
            if key == "images":
                # Keep images as list of PIL Images, maintaining batch order
                collated[key] = []
                for sample in batch:
                    collated[key].extend(sample[key])
                    
            elif key == "inference_type":
                # All items in batch should have same inference type
                collated[key] = elem[key]
                
            elif isinstance(elem[key], (str, int, float, bool)):
                # Basic types go into lists
                collated[key] = [sample[key] for sample in batch]
                
            elif isinstance(elem[key], dict):
                # Handle nested dictionaries (like metadata)
                collated[key] = [sample[key] for sample in batch]
                
            elif isinstance(elem[key], (list, tuple)):
                # Flatten lists/tuples while preserving batch order
                collated[key] = []
                for sample in batch:
                    collated[key].extend(sample[key])
                    
            elif isinstance(elem[key], torch.Tensor):
                # Stack tensors if possible
                try:
                    collated[key] = torch.stack([sample[key] for sample in batch])
                except:
                    collated[key] = [sample[key] for sample in batch]
                    
            elif isinstance(elem[key], np.ndarray):
                # Convert numpy arrays to tensors and stack
                try:
                    collated[key] = torch.stack([torch.from_numpy(sample[key]) for sample in batch])
                except:
                    collated[key] = [sample[key] for sample in batch]
            else:
                # Default to list for unknown types
                collated[key] = [sample[key] for sample in batch]
                
        except Exception as e:
            # If anything fails, fall back to list
            collated[key] = [sample[key] for sample in batch]
            
    return collated

class InferencePipeline:
    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        batch_size: int,
        output_dir: Path,
        config: Dict[str, Any]
    ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.config = config
        self.logger = logging.getLogger("vlm_infer")
        
        # Create organized output directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / dataset.name / model.config.name / dataset.inference_type.value / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save run configuration and details
        self._save_run_info()
    
    def _save_run_info(self,results=None):
        """Save detailed information about the run."""
        if results is not None:
            num_results = len(results)
            num_errors = sum(1 for result in results if result["error"] is not None)
        else:
            num_results = None
            num_errors = None
        
        run_info = {
            "timestamp": datetime.now().isoformat(),
            "model": {
                "name": self.model.config.name,
                "type": self.model.config.type,
                "checkpoint": getattr(self.model.config, "checkpoint", None),
                "max_new_tokens": getattr(self.model.config, "max_new_tokens", None),
                "temperature": getattr(self.model.config, "temperature", None),
                "top_p": getattr(self.model.config, "top_p", None),
                "top_k": getattr(self.model.config, "top_k", None),
                "num_return_sequences": getattr(self.model.config, "num_return_sequences", None),
                "torch_dtype": getattr(self.model.config, "torch_dtype", None),
                "seed": getattr(self.model.config, "seed", None),
                "device": self.model.device,
                "generation_config": self.model.config.to_dict()
            },
            "dataset": {
                "name": self.dataset.name,
                "size": len(self.dataset),
                "inference_type": self.dataset.inference_type.value,
                "version": getattr(self.dataset, "version", None)
            },
            "batch_size": self.batch_size,
            "full_config": self.config.to_dict(),
            "num_results": num_results,
            "num_errors": num_errors
        }
        
        # Save as JSON for easy reading
        with open(self.run_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)
            
        # Save full config as YAML
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)
    
    def run(self) -> List[Dict[str, Any]]:
        """Run inference and save results."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            persistent_workers=True
        )
        
        results = []
        for batch in tqdm(dataloader, desc="Running inference"):
            try:
                # Get model inputs and generate outputs
                model_inputs = self.model.prepare_inputs(batch)
                outputs = self.model.generate(model_inputs["all_inputs"])
                
                # Combine outputs with metadata
                for idx, output in enumerate(outputs):
                    result = {
                        "id": batch["id"][idx],
                        "question": batch["questions"][idx],
                        "answer": output,
                        "metadata": batch.get("metadata", {})[idx] if "metadata" in batch else {},
                        "prompt": model_inputs.get("text", model_inputs.get("prompt", ""))[idx],
                        "error": None
                    }
                    results.append(result)
                    
            except Exception as e:
                self.logger.error(f"Error processing batch: {str(e)}", exc_info=True)
                results.append({
                    "id": batch["id"][idx],
                    "question": batch["questions"][idx],
                    "answer": f"ERROR: {str(e)}",
                    "metadata": batch.get("metadata", {})[idx] if "metadata" in batch else {},
                    "error": str(e)
                })
                continue
        
        # Save results
        self.save_results(results)
        return results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save results with detailed organization."""
        # Save full results with all details
        with open(self.run_dir / "detailed_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save CSV format for easy analysis
        if isinstance(self.dataset, VToMDataset):
            self._save_vtom_results(results)
        else:
            self._save_default_results(results)

        self._save_run_info(results)
    
    def _save_vtom_results(self, results: List[Dict[str, Any]]):
        """Save VToM specific results."""
        df = self.dataset.get_original_data()
        version = self.dataset.version
        inference_type = self.dataset.inference_type.value
        
        # Create model output columns
        model_outputs = {result["id"]: result["answer"] for result in results}
        model_prompts = {result["id"]: result["prompt"] for result in results}

        # Add columns with version and inference type info
        output_col = f"model_generation"
        prompt_col = f"model_prompt"
        
        df[output_col] = df.index.map(lambda x: model_outputs.get(x, ""))
        df[prompt_col] = df.index.map(lambda x: model_prompts.get(x, ""))
        
        # Save CSV
        df.to_csv(self.run_dir / "results.csv", index=False)
    
    def _save_default_results(self, results: List[Dict[str, Any]]):
        """Save results for other datasets."""
        df = pd.DataFrame(results)
        df.to_csv(self.run_dir / "results.csv", index=False) 