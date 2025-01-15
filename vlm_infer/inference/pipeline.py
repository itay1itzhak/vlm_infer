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
        self.output_dir = output_dir
        self.config = config
        self.logger = logging.getLogger("vlm_infer")
        
    def run(self) -> List[Dict[str, Any]]:
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,  # Can use multiple workers
            collate_fn=custom_collate_fn,
            pin_memory=True,  # Better GPU transfer
            persistent_workers=True,  # Keep workers alive between batches
            prefetch_factor=2  # Prefetch next batches
        )
        
        results = []
        for batch in tqdm(dataloader, desc="Running inference"):
            try:                
                outputs = self.model(batch)
                
                # Combine outputs with metadata
                for idx, output in enumerate(outputs):
                    result = {
                        "id": batch["id"][idx],
                        "question": batch["questions"][idx],
                        "answer": output,
                        "metadata": batch.get("metadata", {})[idx] if "metadata" in batch else {}
                    }
                    results.append(result)
                    
            except Exception as e:
                self.logger.error(f"Error processing batch: {str(e)}", exc_info=True)
                results.append({
                    "id": batch["id"][idx],
                    "question": batch["questions"][idx],
                    "answer": str(e),
                    "metadata": batch.get("metadata", {})[idx] if "metadata" in batch else {}
                })
                continue
                
        return results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save results with inference type information."""
        if isinstance(self.dataset, VToMDataset):
            self._save_vtom_results(results)
        else:
            self._save_default_results(results)
    
    def _save_vtom_results(self, results: List[Dict[str, Any]]):
        df = self.dataset.get_original_data()
        version = self.dataset.version
        inference_type = self.dataset.inference_type.value
        
        # Create model output column with version and inference type info
        model_outputs = {result["id"]: result["answer"] for result in results}
        column_name = f"model_generation_{self.model.config.name}_{version}_{inference_type}"
        df[column_name] = df.index.map(lambda x: model_outputs.get(x, ""))
        
        # Save updated CSV
        output_file = self.output_dir / f"results_{version}_{inference_type}.csv"
        df.to_csv(output_file, index=False)
        
        # Save detailed results
        detailed_output = self.output_dir / f"detailed_results_{version}_{inference_type}.json"
        with open(detailed_output, "w") as f:
            json.dump(results, f, indent=2)
    
    def _save_default_results(self, results: List[Dict[str, Any]]):
        """Default JSON results saving."""
        output_file = self.output_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2) 