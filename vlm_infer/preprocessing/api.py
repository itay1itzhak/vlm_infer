from typing import Dict, Any, List
from PIL import Image
import base64
from io import BytesIO
from .base import BasePreprocessor, PreprocessConfig
from .image import ImagePreprocessor

class OpenAIPreprocessor(BasePreprocessor):
    def __init__(self, config: PreprocessConfig):
        super().__init__(config)
        self.image_preprocessor = ImagePreprocessor()
    
    def preprocess_image(self, image: Image.Image) -> str:
        """Convert image to base64 string after preprocessing."""
        if self.config.pad_to_square:
            image = self.image_preprocessor.pad_to_square(image)
            
        image = self.image_preprocessor.resize_image(
            image,
            self.config.image_size,
            self.config.resize_method
        )
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    
    def prepare_inputs(
        self,
        images: List[Image.Image],
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for OpenAI API."""
        # Preprocess images
        image_urls = [self.preprocess_image(img) for img in images]
        
        # Create message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    *[{"type": "image_url", "image_url": {"url": url}} 
                      for url in image_urls]
                ]
            }
        ]
        
        return {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 300),
            "temperature": kwargs.get("temperature", 0.7)
        } 