import pytest
import torch
from pathlib import Path

from vlm_infer.models import get_model
from vlm_infer.models.base import BaseModel
from vlm_infer.models.hf_models import HuggingFaceModel
from vlm_infer.models.api_models import APIModel

@pytest.fixture
def model_config():
    """Create test model config."""
    return {
        "name": "test_model",
        "type": "hf",
        "checkpoint": "test_checkpoint",
        "max_new_tokens": 100,
        "temperature": 0.7
    }

def test_model_factory(model_config):
    """Test model factory function."""
    with pytest.raises(ValueError):
        get_model("nonexistent_model", model_config)

def test_base_model(model_config):
    """Test base model class."""
    class TestModel(BaseModel):
        def load(self):
            pass
        
        def generate(self, inputs):
            return ["test output"]
    
    model = TestModel(model_config)
    assert model.config == model_config
    assert callable(model.generate)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_huggingface_model(model_config):
    """Test HuggingFace model initialization."""
    with pytest.raises(Exception):  # Should fail without actual model
        model = HuggingFaceModel(model_config)
        model.load()

def test_api_model(model_config):
    """Test API model initialization."""
    model_config.type = "openai"
    model_config.api_key = "test_key"
    model_config.api_base = "https://api.test.com"
    
    model = APIModel(model_config)
    assert model.api_key == "test_key"
    assert model.api_base == "https://api.test.com" 