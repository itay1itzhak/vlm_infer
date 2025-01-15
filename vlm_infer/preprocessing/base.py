from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import torch
import torchvision.transforms as T

@dataclass
class PreprocessConfig:
    image_size: tuple = (224, 224)
    resize_method: str = "bilinear"
    normalize: bool = True
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    pad_to_square: bool = False
    apply_color_jitter: bool = False
    convert_to_tensor: bool = True

class BasePreprocessor:
    """Base class for model-specific preprocessors."""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
    
    def preprocess_image(self, image: Image.Image) -> Union[Image.Image, torch.Tensor]:
        """Apply preprocessing pipeline to image."""
        if self.config.pad_to_square:
            image = self.image_preprocessor.pad_to_square(image)
            
        image = self.image_preprocessor.resize_image(
            image,
            self.config.image_size,
            self.config.resize_method
        )
        
        if self.config.convert_to_tensor:
            image = T.ToTensor()(image)
            
            if self.config.normalize:
                image = T.Normalize(
                    mean=self.config.mean,
                    std=self.config.std
                )(image)
        
        return image
    
    def prepare_inputs(
        self,
        images: List[Image.Image],
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for model."""
        raise NotImplementedError 