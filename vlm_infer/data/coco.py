from typing import Dict, Any, List
import json
from pathlib import Path
from PIL import Image
import torch
from .base import BaseDataset

class COCODataset(BaseDataset):
    """COCO VQA dataset implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_dir = Path(config.data_dir)
        self.split = config.split
        self.max_samples = config.max_samples
        self.image_size = config.image_size or (224, 224)
        
        # Load annotations
        self.questions = self._load_questions()
        self.annotations = self._load_annotations()
        
        if self.max_samples:
            self.questions = self.questions[:self.max_samples]
            
    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load VQA questions."""
        question_file = self.data_dir / f"v2_OpenEnded_mscoco_{self.split}2014_questions.json"
        with open(question_file) as f:
            data = json.load(f)
        return data["questions"]
    
    def _load_annotations(self) -> Dict[int, Dict[str, Any]]:
        """Load VQA annotations if available."""
        try:
            anno_file = self.data_dir / f"v2_mscoco_{self.split}2014_annotations.json"
            with open(anno_file) as f:
                data = json.load(f)
            return {item["question_id"]: item for item in data["annotations"]}
        except FileNotFoundError:
            return {}
            
    def _load_image(self, image_id: int) -> Image.Image:
        """Load and preprocess image."""
        image_path = self.data_dir / f"COCO_{self.split}2014_{image_id:012d}.jpg"
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size)
        return image
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        question = self.questions[idx]
        question_id = question["question_id"]
        image_id = question["image_id"]
        
        # Load image
        image = self._load_image(image_id)
        
        # Get annotation if available
        annotation = self.annotations.get(question_id, {})
        
        return {
            "id": question_id,
            "images": [image],  # List for consistency with multi-image cases
            "questions": [question["question"]],
            "metadata": {
                "image_id": image_id,
                "ground_truth": annotation.get("multiple_choice_answer", ""),
                "answers": annotation.get("answers", [])
            }
        }
    
    def __len__(self) -> int:
        return len(self.questions) 