from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, Tuple
from copy import deepcopy
import yaml
from pathlib import Path
from .types import InferenceType
from .configs.defaults import load_default_config

@dataclass
class InferenceConfig:
    """Inference-specific configuration."""
    batch_size: int = 1
    num_workers: int = 4
    save_images: bool = False
    verbose: bool = True
    save_dir: Optional[str] = None
    save_dir_path: Optional[Path] = None
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "save_images": self.save_images,
            "verbose": self.verbose,
            "save_dir": self.save_dir,
            "save_dir_path": self.save_dir_path
        }
    
@dataclass
class ModelConfig:
    """Model-specific configuration."""
    name: str
    type: str = field(default="hf")
    
    # Model loading settings
    checkpoint: Optional[str] = None
    model_path: Optional[str] = None
    base_model_type: Optional[str] = None
    model_class: Optional[str] = None
    
    # API settings
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    num_beams: int = 1
    do_sample: bool = False
    top_k: int = 50
    num_return_sequences: int = 1
    seed: int = 42
    
    # Model behavior settings
    is_chat_model: bool = True
    prompt_template: Optional[str] = None
    
    # Hardware settings
    torch_dtype: str = "float16"
    device_map: str = "auto"
    
    def __post_init__(self):
        valid_types = ["hf", "openai", "local", "api"]
        if self.type not in valid_types:
            raise ValueError(f"Invalid model type: {self.type}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "checkpoint": self.checkpoint,
            "model_path": self.model_path,
            "base_model_type": self.base_model_type,
            "model_class": self.model_class
        }

@dataclass
class DatasetConfig:
    """Dataset-specific configuration."""
    name: str
    
    # Path settings
    data_dir: Optional[str] = None
    dataset_path: Optional[str] = None
    
    # Dataset specific settings
    version: Optional[str] = None
    split: str = "test"
    csv_file: Optional[str] = None
    
    # Data loading settings
    max_samples: Optional[int] = None
    image_size: Tuple[int, int] = (224, 224)
    
    # Inference settings
    inference_type: Optional[str] = None
    
    def __post_init__(self):
        if not self.data_dir and not self.dataset_path:
            raise ValueError("Either data_dir or dataset_path must be provided")
        
        # Validate inference_type if provided
        if self.inference_type:
            try:
                InferenceType(self.inference_type)
            except ValueError:
                valid_types = [t.value for t in InferenceType]
                raise ValueError(
                    f"Invalid inference_type '{self.inference_type}'. "
                    f"Must be one of: {', '.join(valid_types)}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "data_dir": self.data_dir,
            "dataset_path": self.dataset_path
        }

@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    dataset: DatasetConfig
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        # Extract config sections
        model_dict = config_dict.get("model", {})
        dataset_dict = config_dict.get("dataset", {})
        inference_dict = config_dict.get("inference", {})
        
        # Create sub-configs
        model_config = ModelConfig(**model_dict)
        dataset_config = DatasetConfig(**dataset_dict)
        inference_config = InferenceConfig(**inference_dict)
        
        return cls(
            model=model_config,
            dataset=dataset_config,
            inference=inference_config
        )
    
    @classmethod
    def from_yaml(cls, config_path: str, dataset_path: str = None, inference_type: str = None) -> 'Config':
        """Load config from YAML with optional overrides.
        
        Args:
            config_path: Path to YAML config file
            dataset_path: Optional override for dataset path
            inference_type: Optional override for inference type
            
        Returns:
            Config object with all settings
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Override dataset path if provided
        if dataset_path:
            if 'dataset' not in config_dict:
                config_dict['dataset'] = {}
            config_dict['dataset']['dataset_path'] = dataset_path
        
        # Override inference type if provided via CLI
        if inference_type:
            if 'dataset' not in config_dict:
                config_dict['dataset'] = {}
            config_dict['dataset']['inference_type'] = inference_type
            
        # Use from_dict to properly create nested configs
        return cls.from_dict(config_dict)
    
    @staticmethod
    def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two config dictionaries."""
        merged = deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = Config._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged 
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model.to_dict(),
            "dataset": self.dataset.to_dict(),
            "inference": self.inference.to_dict()
        }