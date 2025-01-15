import pytest
from pathlib import Path
import pandas as pd
from PIL import Image

from vlm_infer.data.vtom import VToMDataset
from .utils import create_test_image, create_test_dataset

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary test data directory."""
    data_dir = tmp_path / "take_4.1"
    data_dir.mkdir(parents=True)
    
    # Create test CSV
    df = create_test_dataset()
    df.to_csv(data_dir / "take_4.1.csv", index=False)
    
    # Create test images
    img_dir = data_dir / "generated_images" / "story_1"
    img_dir.mkdir(parents=True)
    test_image = create_test_image()
    test_image.save(img_dir / "combined.png")
    
    return tmp_path

def test_vtom_dataset_initialization(test_data_dir):
    """Test VToM dataset initialization."""
    config = {
        "data_dir": str(test_data_dir),
        "version": "take_4.1"
    }
    
    dataset = VToMDataset(config)
    assert len(dataset) == 5
    assert "story_structure" in dataset.data.columns
    assert "processed_image_path" in dataset.data.columns

def test_vtom_dataset_getitem(test_data_dir):
    """Test VToM dataset item retrieval."""
    config = {
        "data_dir": str(test_data_dir),
        "version": "take_4.1"
    }
    
    dataset = VToMDataset(config)
    item = dataset[0]
    
    assert "id" in item
    assert "images" in item
    assert "questions" in item
    assert "metadata" in item
    assert isinstance(item["images"][0], Image.Image)
    assert isinstance(item["questions"][0], str)

def test_vtom_dataset_invalid_path():
    """Test VToM dataset with invalid path."""
    config = {
        "data_dir": "nonexistent_path",
        "version": "take_4.1"
    }
    
    with pytest.raises(FileNotFoundError):
        VToMDataset(config) 