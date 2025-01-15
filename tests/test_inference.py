import pytest
from pathlib import Path
import torch

from vlm_infer.inference import InferencePipeline
from vlm_infer.models.base import BaseModel
from vlm_infer.data.base import BaseDataset
from .utils import create_test_dataset, create_test_image

class TestModel(BaseModel):
    def load(self):
        pass
    
    def generate(self, inputs):
        return ["test answer"] * len(inputs["questions"])

class TestDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.data = create_test_dataset()
    
    def __getitem__(self, idx):
        return {
            "id": idx,
            "images": [create_test_image()],
            "questions": ["test question"],
            "metadata": {"test": "metadata"}
        }
    
    def __len__(self):
        return len(self.data)

@pytest.fixture
def test_pipeline(tmp_path):
    """Create test inference pipeline."""
    model = TestModel({"name": "test_model", "type": "test"})
    dataset = TestDataset({"name": "test_dataset"})
    
    return InferencePipeline(
        model=model,
        dataset=dataset,
        batch_size=2,
        output_dir=tmp_path,
        config={}
    )

def test_pipeline_run(test_pipeline):
    """Test inference pipeline run."""
    results = test_pipeline.run()
    assert len(results) > 0
    assert "id" in results[0]
    assert "answer" in results[0]
    assert "metadata" in results[0]

def test_pipeline_save_results(test_pipeline):
    """Test results saving."""
    results = test_pipeline.run()
    test_pipeline.save_results(results)
    
    assert (test_pipeline.output_dir / "results.json").exists() 