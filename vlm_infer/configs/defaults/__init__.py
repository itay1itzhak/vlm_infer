from pathlib import Path
import yaml
from typing import Dict, Any

CONFIGS_DIR = Path(__file__).parent

def load_default_config(model_name: str) -> Dict[str, Any]:
    """Load default configuration for a given model."""
    config_path = CONFIGS_DIR / f"{model_name}.yaml"
    if not config_path.exists():
        # Try to load based on model type
        model_type = model_name.split("-")[0]  # e.g., llava, gpt4v
        config_path = CONFIGS_DIR / f"{model_type}.yaml"

    # print files in config dir
    print(f"Files in config dir: {CONFIGS_DIR}")
    for file in CONFIGS_DIR.iterdir():
        print(file)
    
    if not config_path.exists():
        raise ValueError(f"No default config found for model: {model_name}")
        
    with open(config_path) as f:
        return yaml.safe_load(f) 