from typing import Dict, Any
from .base import BaseDataset
from .coco import COCODataset
from .mmbench import MMBenchDataset
from .vtom import VToMDataset

DATASET_REGISTRY = {
    "coco": COCODataset,
    "mmbench": MMBenchDataset,
    "vtom": VToMDataset
}

def get_dataset(dataset_name: str, config: Dict[str, Any]) -> BaseDataset:
    """Factory function to create datasets."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {dataset_name} not supported. Available datasets: {list(DATASET_REGISTRY.keys())}")
    
    dataset_cls = DATASET_REGISTRY[dataset_name]
    return dataset_cls(config) 