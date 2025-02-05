from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Optional
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from ..types import InferenceType

# Create a Protocol for config instead of importing the class
from typing import Protocol

class DatasetConfigProtocol(Protocol):
    """Protocol defining required dataset config interface."""
    name: str
    data_dir: Optional[str]
    dataset_path: Optional[str]
    inference_type: Optional[str]

class BaseDataset(Dataset, ABC):
    """Base class for all datasets."""
    
    def __init__(self, config: DatasetConfigProtocol):
        """Initialize dataset with config."""
        self.config = config
        
        # Set default inference type if none provided
        inference_type = config.inference_type or "standard"
        
        # Validate and set inference type
        try:
            self.inference_type = InferenceType(inference_type)
        except ValueError:
            valid_types = [t.value for t in InferenceType]
            raise ValueError(
                f"Invalid inference_type '{inference_type}'. "
                f"Must be one of: {', '.join(valid_types)}"
            )
        
        # Set dataset name
        self.name = self._get_dataset_name()
    
    def _get_dataset_name(self) -> str:
        """Get dataset name from config or dataset path."""
        # If dataset_path is provided, use the file name without extension
        if hasattr(self.config, "dataset_path") and self.config.dataset_path:
            return Path(self.config.dataset_path).stem
        
        # Otherwise use the default name from the dataset class
        return self.default_name
    
    @property
    @abstractmethod
    def default_name(self) -> str:
        """Default name for the dataset."""
        pass
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item based on inference type."""
        item = self._get_base_item(idx)
        item["inference_type"] = self.inference_type.value
        return item
    
    @abstractmethod
    def _get_base_item(self, idx: int) -> Dict[str, Any]:
        """Get base item implementation."""
        pass
    
    @abstractmethod
    def load_data(self):
        """Load dataset metadata."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset."""
        pass 