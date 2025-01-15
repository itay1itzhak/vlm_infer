from typing import Dict, Any, List
from PIL import Image
import torch
from .base import BasePreprocessor, PreprocessConfig
from .image import ImagePreprocessor

class LlamaPreprocessor(BasePreprocessor):
    def __init__(self, config: PreprocessConfig, processor: Any):
        super().__init__(config)
        self.processor = processor
        self.image_preprocessor = ImagePreprocessor()
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Apply Llama-specific preprocessing."""
        if self.config.pad_to_square:
            image = self.image_preprocessor.pad_to_square(image)
            
        image = self.image_preprocessor.resize_image(
            image,
            self.config.image_size,
            self.config.resize_method
        )
        
        if self.config.normalize:
            image = self.image_preprocessor.normalize(
                image,
                mean=self.config.mean,
                std=self.config.std
            )
            
        return image
    
    def prepare_inputs(
        self,
        images: List[Image.Image],
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for Llama model."""
        # Preprocess images
        processed_images = [self.preprocess_image(img) for img in images]
        
        # Create messages format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text}
                ]
            }
        ]
        
        # Apply chat template
        input_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            processed_images[0],  # Llama expects single image
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        return inputs 