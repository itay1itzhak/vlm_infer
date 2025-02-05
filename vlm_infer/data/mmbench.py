from typing import Dict, Any, List
import json
import pandas as pd
from pathlib import Path
from PIL import Image
from .base import BaseDataset
from vlm_infer.config import DatasetConfig

class MMBenchDataset(BaseDataset):
    """MMBench dataset implementation."""
    
    default_name = "mmbench"
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.data_dir = Path(config.data_dir)
        self.split = config.split
        self.max_samples = config.max_samples
        self.image_size = config.image_size
        
        # Load data
        self.data = self._load_data()
        if self.max_samples:
            self.data = self.data[:self.max_samples]
            
    def _load_data(self) -> pd.DataFrame:
        """Load MMBench data."""
        csv_path = self.data_dir / f"mmbench_{self.split}.csv"
        df = pd.read_csv(csv_path)
        return df
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        full_path = self.data_dir / "images" / image_path
        image = Image.open(full_path).convert("RGB")
        image = image.resize(self.image_size)
        return image
    
    def _format_question(self, row: pd.Series) -> str:
        """Format question with options."""
        question = row["question"]
        options = [row[f"option_{i}"] for i in range(4) if pd.notna(row[f"option_{i}"])]
        
        formatted_question = f"{question}\n"
        for i, opt in enumerate(options):
            formatted_question += f"{chr(65+i)}. {opt}\n"
        
        return formatted_question.strip()
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        row = self.data.iloc[idx]
        
        # Load image(s)
        images = []
        if pd.notna(row["image"]):
            images.append(self._load_image(row["image"]))
        if pd.notna(row.get("image_2", None)):  # Handle multiple images if present
            images.append(self._load_image(row["image_2"]))
            
        # Format question with options
        question = self._format_question(row)
        
        return {
            "id": row["index"],
            "images": images,
            "questions": [question],
            "metadata": {
                "ground_truth": row.get("answer", ""),
                "category": row.get("category", ""),
                "l2_category": row.get("l2_category", "")
            }
        }
    
    def __len__(self) -> int:
        return len(self.data) 