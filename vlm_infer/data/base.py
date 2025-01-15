from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator
from torch.utils.data import Dataset
from PIL import Image
from vlm_infer.types import InferenceType

class BaseDataset(Dataset, ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.inference_type = InferenceType(config.inference_type)
        
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