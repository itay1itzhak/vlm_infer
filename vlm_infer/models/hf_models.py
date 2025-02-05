from typing import List, Dict, Any
import torch
from transformers import (
    AutoProcessor, AutoModelForCausalLM, 
    MllamaForConditionalGeneration,
    LlavaForConditionalGeneration
)

from vlm_infer.config import ModelConfig
from .base import BaseModel
from vlm_infer.utils.env import get_cache_dir, get_api_key
from vlm_infer.preprocessing.base import PreprocessConfig
from vlm_infer.preprocessing.llava import LlavaPreprocessor
from vlm_infer.preprocessing.llama import LlamaPreprocessor

class HuggingFaceModel(BaseModel):
    """Base class for Hugging Face models."""
    def __init__(self, config: ModelConfig, device: str = "cuda"):
        super().__init__(config, device)
        self.hf_token = get_api_key("huggingface")
        self.model = None
        self.processor = None
        self.preprocessor = None
        
        # Use direct attribute access for dataclass
        self.preprocess_config = PreprocessConfig(
            image_size=getattr(config, "image_size", (224, 224)),
            resize_method=getattr(config, "resize_method", "bilinear"),
            normalize=getattr(config, "normalize", True),
            pad_to_square=getattr(config, "pad_to_square", False),
            apply_color_jitter=getattr(config, "apply_color_jitter", False)
        )
        self.load()

    def generate(self, inputs: Dict[str, Any]) -> List[str]:
        """Generate responses using HuggingFace model with reproducible results."""
        if not self.model or not self.processor:
            raise ValueError("Model or processor not initialized")
            
        # Set random seed for reproducibility
        torch.manual_seed(42)  # Default seed for reproducibility
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            
        # Get generation config from model config using direct attribute access
        generation_config = {
            "max_new_tokens": getattr(self.config, "max_new_tokens", 512),
            "temperature": getattr(self.config, "temperature", 0.0),  # 0 for greedy
            "top_p": getattr(self.config, "top_p", 1.0),  # 1.0 for no nucleus sampling
            "do_sample": getattr(self.config, "do_sample", False),  # False for greedy
            "num_beams": getattr(self.config, "num_beams", 1),  # 1 for greedy
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        # If temperature > 0, enable sampling
        if generation_config["temperature"] > 0:
            generation_config["do_sample"] = True
        
        # Generate outputs
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode outputs
        if hasattr(self.processor, "batch_decode"):
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
        else:
            decoded = [
                self.processor.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
        # Clean up responses (remove prompt if present)
        cleaned = []
        for output in decoded:
            # Remove any system prompt or instruction
            if "<|assistant|>" in output or "ASSISTANT:" in output:
                output = output.split("<|assistant|>")[-1]
                output = output.split("ASSISTANT:")[-1]
            elif "ASSISTANT:" in output:
                output = output.split("ASSISTANT:")[-1]
    
            cleaned.append(output.strip())
            
        return cleaned

    def prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs using model-specific preprocessor."""
        if not self.preprocessor:
            raise ValueError("Preprocessor not initialized")
        
        # Get batch size from questions
        batch_size = len(batch["questions"])
        all_prompts = []
        
        # Process each sample in the batch
        for i in range(batch_size):
            prompt_parts = []
            
            # Add story if present
            if "stories" in batch:
                prompt_parts.append(f"Story: {batch['stories'][i]}")
            
            # Add details if present
            if "details" in batch:
                details = batch["details"][i]
                if isinstance(details, dict):
                    details_text = "\n".join(f"{k.replace('_', ' ').title()}: {v}" 
                                           for k, v in details.items())
                    prompt_parts.append(f"Story Details:\n{details_text}")
            
            # Add captions if present
            if "captions" in batch:
                captions = batch["captions"][i]
                if isinstance(captions, dict):
                    caption_text = "\n".join(f"{k}: {v}" for k, v in captions.items())
                    prompt_parts.append(f"Context:\n{caption_text}")
                else:
                    caption_text = str(captions)
                    prompt_parts.append(f"Context:\n{caption_text}")
            
            # Add question
            text = batch.get("questions", [""] * batch_size)[i]
            prompt_parts.append(f"Question: {text}")
            
            # Combine all parts
            full_text = "\n".join(prompt_parts)
            
            # Create conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_text}
                    ],
                },
            ]
            
            # Add image if present
            if "images" in batch and batch["images"]:
                conversation[0]["content"].append({"type": "image"})
            
            # Apply chat template
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )
            all_prompts.append(prompt)

        # Process all images and prompts together
        inputs = self.processor(
            images=batch.get("images", None),
            text=all_prompts,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs["all_inputs"] = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["prompt"] = all_prompts
        
        return inputs

    def __call__(self, batch: Dict[str, Any]) -> List[str]:
        inputs = self.prepare_inputs(batch)
        return self.generate(inputs)

class LlavaModel(HuggingFaceModel):
    """Handler for LLaVA models."""
    def load(self):
        checkpoint = self.config.checkpoint
        cache_dir = get_cache_dir()
        
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, 
            cache_dir=cache_dir, 
            token=self.hf_token
        )
        
        print(f"Loading model {checkpoint} from cache dir {cache_dir}")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            token=self.hf_token
        )
        
        # Initialize preprocessor
        self.preprocessor = LlavaPreprocessor(
            config=self.preprocess_config,
            processor=self.processor
        )

class LlamaVisionModel(HuggingFaceModel):
    """Handler for Llama Vision models."""
    def load(self):
        checkpoint = self.config.checkpoint
        cache_dir = get_cache_dir()
        
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, 
            cache_dir=cache_dir, 
            token=self.hf_token
        )
        
        self.model = MllamaForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            token=self.hf_token
        )
        
        # Initialize preprocessor
        self.preprocessor = LlamaPreprocessor(
            config=self.preprocess_config,
            processor=self.processor
        )

class LocalModel(HuggingFaceModel):
    """Handler for locally fine-tuned models."""
    def load(self):
        local_path = self.config.model_path
        model_type = self.config.get("base_model_type", "llava")
        
        self.processor = AutoProcessor.from_pretrained(local_path, token=self.hf_token)
        
        if model_type == "llava":
            model_class = LlavaForConditionalGeneration
        elif model_type == "llama-vision":
            model_class = MllamaForConditionalGeneration
        else:
            raise ValueError(f"Unsupported base model type: {model_type}")
            
        self.model = model_class.from_pretrained(
            local_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=self.hf_token
        ) 