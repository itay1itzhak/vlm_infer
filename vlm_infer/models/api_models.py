from typing import List, Dict, Any
from openai import OpenAI
from .base import BaseModel
from vlm_infer.utils.env import get_api_key
from vlm_infer.preprocessing.base import PreprocessConfig
from vlm_infer.preprocessing.api import OpenAIPreprocessor
import os

class APIModel(BaseModel):
    """Base class for API-based models."""
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        super().__init__(config, device)
        
        # Get API key from environment if not in config
        self.api_key = config.get("api_key") or get_api_key("openai")
        self.api_base = config.get("api_base") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize preprocessor config
        self.preprocess_config = PreprocessConfig(
            image_size=config.get("image_size", (224, 224)),
            resize_method=config.get("resize_method", "bilinear"),
            pad_to_square=config.get("pad_to_square", True)
        )
        
        self.preprocessor = None

class OpenAIModel(APIModel):
    """Handler for OpenAI GPT-4V models."""
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        super().__init__(config, device)
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = "gpt-4-vision-preview" if "gpt4v" in config.name else config.name
        self.preprocessor = OpenAIPreprocessor(self.preprocess_config)
    
    def prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs using OpenAI preprocessor."""
        images = batch["images"]
        text = batch.get("questions", [""])[0]
        
        # Add story if present
        if "stories" in batch:
            text = f"Story: {batch['stories'][0]}\nQuestion: {text}"
            
        # Add captions if present
        if "captions" in batch:
            captions = batch["captions"][0]
            if isinstance(captions, dict):
                caption_text = "\n".join(f"{k}: {v}" for k, v in captions.items())
            else:
                caption_text = str(captions)
            text = f"Context:\n{caption_text}\nQuestion: {text}"
        
        return self.preprocessor.prepare_inputs(
            images,
            text,
            max_tokens=self.config.get("max_new_tokens", 300),
            temperature=self.config.get("temperature", 0.7)
        )
    
    def generate(self, inputs: Dict[str, Any]) -> List[str]:
        """Generate responses using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=inputs["messages"],
            max_tokens=inputs["max_tokens"],
            temperature=inputs["temperature"]
        )
        
        return [response.choices[0].message.content] 