from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from copy import deepcopy
from .configs.defaults import load_default_config
from vlm_infer.utils.env import get_cache_dir

def default_preprocessing() -> Dict[str, Any]:
    return {
        "resize": True,
        "normalize": True
    }

@dataclass
class ModelConfig:
    # Required fields
    name: str
    type: str = field(default="hf")  # Default to HuggingFace models
    
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
    
    # Model behavior settings
    is_chat_model: bool = True
    prompt_template: Optional[str] = None
    
    # Hardware settings
    torch_dtype: str = "float16"
    device_map: str = "auto"
    
    def __post_init__(self):
        # Validate model type
        valid_types = ["hf", "openai", "local", "api"]
        if self.type not in valid_types:
            raise ValueError(f"Invalid model type: {self.type}. Must be one of {valid_types}")
        
        # Validate required fields based on type
        if self.type == "hf" and not (self.checkpoint or self.model_path):
            raise ValueError("HuggingFace models require either checkpoint or model_path")
        
        if self.type == "local" and not self.model_path:
            raise ValueError("Local models require model_path")
        
        # Validate numeric parameters
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if not 0 <= self.top_p <= 1:
            raise ValueError("Top-p must be between 0 and 1")
        
        # Validate torch dtype
        valid_dtypes = ["float16", "float32", "bfloat16"]
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(f"Invalid torch_dtype: {self.torch_dtype}. Must be one of {valid_dtypes}")

@dataclass
class DatasetConfig:
    # Required fields
    name: str = "vtom"  # Default to VToM dataset
    
    # Dataset path configuration (mutually exclusive)
    dataset_path: Optional[str] = None  # Direct path to CSV file
    data_dir: Optional[str] = None      # Base directory containing all versions
    
    # Dataset organization
    version: str = "take_4.1"           # Dataset version (e.g., "take_4.1")
    csv_file: Optional[str] = None      # CSV filename (defaults to {version}.csv)
    split: str = "test"
    
    # Data loading settings
    max_samples: Optional[int] = None
    batch_size: int = 1
    num_workers: int = 4
    
    # Image processing settings
    image_size: tuple = (224, 224)
    preprocessing: Dict[str, Any] = field(default_factory=default_preprocessing)
    
    # Inference settings
    inference_type: str = "standard"
    
    def __post_init__(self):
        # Validate path configuration
        if not self.dataset_path and not self.data_dir:
            raise ValueError("Either dataset_path or data_dir must be provided")
            
        # Set default csv_file if not provided
        if not self.csv_file and self.version:
            self.csv_file = f"{self.version}.csv"
            
        # Validate image size
        if not isinstance(self.image_size, (tuple, list)) or len(self.image_size) != 2:
            raise ValueError("image_size must be a tuple/list of (height, width)")
        
        # Validate numeric parameters
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        
        # Validate split
        valid_splits = ["train", "val", "test"]
        if self.split not in valid_splits:
            raise ValueError(f"Invalid split: {self.split}. Must be one of {valid_splits}")
            
        # Validate inference type
        valid_inference_types = ["standard", "image_story", "story_only", 
                               "image_captions", "captions_only", "story_captions"]
        if self.inference_type not in valid_inference_types:
            raise ValueError(f"Invalid inference_type: {self.inference_type}")

class Config:
    def __init__(self, config_dict: Dict[str, Any], dataset_path: Optional[str] = None):
        # Load default config based on model name if specified
        model_name = config_dict.get("model", {}).get("name")
        if model_name:
            default_config = load_default_config(model_name)
            # Merge with user config (user config takes precedence)
            merged_config = deepcopy(default_config)
            merged_config.update(config_dict)
            config_dict = merged_config
        
        # Ensure required sections exist
        if "model" not in config_dict:
            raise ValueError("Config must contain 'model' section")
        if "dataset" not in config_dict:
            config_dict["dataset"] = {}
        
        # Override dataset path if provided
        if dataset_path:
            config_dict["dataset"]["data_dir"] = dataset_path
            
        # Add cache directory to config if not specified
        if "cache_dir" not in config_dict:
            config_dict["cache_dir"] = str(get_cache_dir())
        
        # Create configs with validation
        try:
            self.model_config = ModelConfig(**config_dict["model"])
            self.dataset_config = DatasetConfig(**config_dict["dataset"])
        except TypeError as e:
            raise ValueError(f"Invalid config structure: {str(e)}")
        
        # Store additional inference settings
        self.inference_config = config_dict.get("inference", {
            "batch_size": 1,
            "num_workers": 4,
            "save_images": False,
            "verbose": True
        }) 