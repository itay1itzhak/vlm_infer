from typing import Dict, Any, List
from PIL import Image
import torch
from .base import BasePreprocessor, PreprocessConfig
from .image import ImagePreprocessor

class LlavaPreprocessor(BasePreprocessor):
    def __init__(self, config: PreprocessConfig, processor: Any):
        super().__init__(config)
        self.processor = processor
        self.image_preprocessor = ImagePreprocessor()
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Apply Llava-specific preprocessing."""
        if self.config.pad_to_square:
            image = self.image_preprocessor.pad_to_square(image)
            
        image = self.image_preprocessor.resize_image(
            image,
            self.config.image_size,
            self.config.resize_method
        )
        
        if self.config.apply_color_jitter:
            image = self.image_preprocessor.apply_color_jitter(image)
            
        return image
    
    def prepare_inputs(
        self,
        images: List[Image.Image],
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for Llava model."""
        # Preprocess images
        processed_images = [self.preprocess_image(img) for img in images]
        
        # Create conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image"}
                ]
            }
        ]
        
        # Apply chat template
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            images=processed_images,
            text=prompt,
            return_tensors="pt"
        )
        
        return inputs 