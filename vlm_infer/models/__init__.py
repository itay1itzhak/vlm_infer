from typing import Dict, Any
from .base import BaseModel
from .hf_models import LlamaVisionModel, LlavaModel, LocalModel
from .api_models import OpenAIModel

MODEL_REGISTRY = {
    # LLaMA Vision models
    "llama3.2_11B": LlamaVisionModel,
    "llama3.2_70B": LlamaVisionModel,
    "meta-llama/Llama-3.2-11B-Instruct": LlamaVisionModel,
    "meta-llama/Llama-3.2-70B-Instruct": LlamaVisionModel,

    
    # LLaVA models
    "llava_7B": LlavaModel,
    "llava_13B": LlavaModel,
    "llava-hf/llava-1.5-7b-hf": LlavaModel,
    "llava-hf/llava-1.5-13b-hf": LlavaModel,
    
    # CogVLM models
    #"cogvlm": CogVLMModel,
    
    # OpenAI models
    "gpt4v": OpenAIModel,
    
    # Local models
    "local": LocalModel
}

def get_model(model_name: str, config: Dict[str, Any], **kwargs) -> BaseModel:
    """Get model instance by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(config, **kwargs) 