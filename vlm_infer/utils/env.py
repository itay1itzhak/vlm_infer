import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

def load_env_vars(env_file: Optional[str] = None):
    """Load environment variables from .env file."""
    # Try to load from specified file
    if env_file and os.path.exists(env_file):
        load_dotenv(env_file)
        return

    # Try different locations in order
    project_root = Path(__file__).parent.parent.parent
    possible_locations = [
        project_root / ".env",                    # /code/.env
        project_root / "vlm_infer" / ".env",      # /code/vlm_infer/.env
        Path.home() / ".vlm_infer.env"            # ~/.vlm_infer.env
    ]
    
    for env_path in possible_locations:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment variables from {env_path}")
            return

    print("Warning: No .env file found in any of the following locations:")
    for loc in possible_locations:
        print(f"  - {loc}")

def get_api_key(service: str) -> str:
    """Get API key for specified service."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "huggingface": "HF_TOKEN",
    }
    
    if service not in key_map:
        raise ValueError(f"Unknown service: {service}")
        
    key = os.getenv(key_map[service])
    if not key:
        raise ValueError(f"Missing API key for {service}. Please set {key_map[service]}")
    
    return key

def get_cache_dir() -> Path:
    """Get HuggingFace cache directory."""
    # print current working directory
    print(f"Current working directory: {os.getcwd()}")
    cache_dir = os.getenv("HF_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    
    return Path.home() / ".cache" / "huggingface" 