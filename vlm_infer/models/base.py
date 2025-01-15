from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from pathlib import Path
import torch
from vlm_infer.types import InferenceType

class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        self.config = config
        self.device = device
        
    @abstractmethod
    def load(self):
        """Load the model."""
        pass
    
    def prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs based on inference type."""
        inference_type = InferenceType(batch.get("inference_type", ["standard"])[0])
        
        if inference_type == InferenceType.STANDARD:
            return self._prepare_standard_inputs(batch)
        elif inference_type == InferenceType.IMAGE_STORY:
            return self._prepare_image_story_inputs(batch)
        elif inference_type == InferenceType.STORY_ONLY:
            return self._prepare_story_inputs(batch)
        elif inference_type == InferenceType.IMAGE_CAPTIONS:
            return self._prepare_image_captions_inputs(batch)
        elif inference_type == InferenceType.CAPTIONS_ONLY:
            return self._prepare_captions_inputs(batch)
        elif inference_type == InferenceType.STORY_CAPTIONS:
            return self._prepare_story_captions_inputs(batch)
        else:
            raise ValueError(f"Unsupported inference type: {inference_type}")
    
    def _prepare_standard_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Default implementation for standard inputs."""
        raise NotImplementedError
    
    def _prepare_image_story_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for image + story."""
        raise NotImplementedError
    
    @abstractmethod
    def generate(self, batch: Dict[str, Any]) -> List[str]:
        """Generate responses for a batch of inputs."""
        pass
    
    def __call__(self, batch: Dict[str, Any]) -> List[str]:
        inputs = self.prepare_inputs(batch)
        return self.generate(inputs) 