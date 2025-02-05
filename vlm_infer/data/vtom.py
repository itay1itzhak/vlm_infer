from typing import Dict, Any, List, Union
import pandas as pd
from pathlib import Path
from PIL import Image
import os
from .base import BaseDataset
from vlm_infer.types import InferenceType
from vlm_infer.config import DatasetConfig
import torch

class VToMDataset(BaseDataset):
    """Visual Theory of Mind (VToM) dataset implementation."""
    
    default_name = "vtom"  # Default name when no specific dataset path is provided
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        
        # First try direct dataset path
        if config.dataset_path:
            dataset_path = Path(config.dataset_path)
            if not dataset_path.is_file() or not str(dataset_path).endswith('.csv'):
                raise ValueError(f"Invalid dataset_path: {dataset_path}. Must be a CSV file.")
            
            self.csv_path = dataset_path
            self.version_dir = self.csv_path.parent
            self.version = config.version
            
        # Otherwise use data_dir + version structure
        else:
            if not config.data_dir:
                raise ValueError("Either dataset_path or data_dir must be provided")
                
            self.data_dir = Path(config.data_dir)
            self.version = config.version
            self.csv_file = config.csv_file
            
            # Check if data_dir points directly to version directory
            if (self.data_dir / self.csv_file).exists():
                self.version_dir = self.data_dir
            else:
                self.version_dir = self.data_dir / self.version
                
            if not self.version_dir.exists():
                raise FileNotFoundError(f"VToM version directory not found: {self.version_dir}")
                
            self.csv_path = self.version_dir / self.csv_file
            
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
        # Store other config values
        self.split = config.split
        self.max_samples = config.max_samples
        self.image_size = config.image_size
        
        # Load data
        self.load_data()
        if self.max_samples:
            self.data = self.data[:self.max_samples]

    def load_data(self):
        """Implement abstract method from BaseDataset."""
        self.data = pd.read_csv(self.csv_path)

        # Remove rows where there is no image path
        self.data = self.data[self.data['combined_image_path'].notna()]
        
        # Process image paths
        def process_image_path(path: str) -> str:
            """Convert absolute or relative paths to be relative to version directory."""
            path = Path(path)
            try:
                # Trim the path to the last 4 parts
                rel_path = Path(*path.parts[-3:])
                return str(rel_path)
            except ValueError:
                # If the path is already relative or has a different base
                # Take the part after 'generated_images'
                # Find the index of 'generated_images'
                generated_images_idx = path.parts.index('generated_images')
                # Take the part after 'generated_images'
                return str( Path(*path.parts[generated_images_idx:]))
        
        self.data['processed_image_path'] = self.data['combined_image_path'].apply(process_image_path)
        
        # Validate that all image files exist
        missing_images = []
        for idx, row in self.data.iterrows():
            img_path = self.version_dir / row['processed_image_path']
            if not img_path.exists():
                missing_images.append((idx, str(img_path)))
        
        if missing_images:
            print(f"Warning: {len(missing_images)} images not found out of {len(self.data)}:")
            for idx, path in missing_images[:5]:  # Show first 5 missing images
                print(f"  Row {idx}: {path}")
            if len(missing_images) > 5:
                print(f"  ... and {len(missing_images) - 5} more")
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        full_path = self.version_dir / image_path
        if not full_path.exists():
            print(f"Image not found: {full_path}")
            self.logger.warning(f"Image not found: {full_path}")
            # Create an empty PIL Image instead of tensor
            return Image.new('RGB', self.image_size, (0, 0, 0))
            
        image = Image.open(full_path).convert("RGB")
        image = image.resize(self.image_size)
        return image
    
    def _format_prompt(self, row: pd.Series) -> str:
        """Format prompt with story and question."""
        story = row["story_structure"]
        question = row["question"]
        return f"Story: {story}\nQuestion: {question}"
    
    def _get_base_item(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        
        # Base metadata
        metadata = {
            "story": row["story_structure"],
            "original_question": row["question"],
            "ground_truth": row["expected_answer"],
            "image_path": row["processed_image_path"],
            "version": self.version
        }
        
        # Generate details if needed
        details = None
        if any(self.inference_type == t for t in [InferenceType.STORY_DETAILS]):
            details = {
                "story_type": row.get("story_type", ""),
                "num_people": row.get("num_people", ""),
                "context": row.get("context", ""),
                "scene": row.get("scene_description", ""),
                "character_names": row.get("character_names", ""),
                "character_roles": row.get("character_roles", ""),
                "location": row.get("location", ""),
                "time": row.get("time", ""),
                "emotions": row.get("emotions", ""),
                "actions": row.get("actions", "")
            }
            # Remove empty details
            details = {k: v for k, v in details.items() if v}
        
        # Generate captions if needed
        captions = None
        if any(self.inference_type == t for t in [
            InferenceType.IMAGE_CAPTIONS,
            InferenceType.CAPTIONS_ONLY,
            InferenceType.STORY_CAPTIONS
        ]):
            captions = {
                "story_type": row.get("story_type", ""),
                "num_people": f"Number of people: {row.get('num_people', '')}",
                "context": row.get("context", ""),
                "scene": row.get("scene_description", "")
            }
            # Remove empty captions
            captions = {k: v for k, v in captions.items() if v}
        
        # Prepare data based on inference type
        if self.inference_type == InferenceType.STANDARD:
            return {
                "id": idx,
                "images": [self._load_image(row["processed_image_path"])],
                "questions": [row["question"]],
                "metadata": metadata
            }
        
        elif self.inference_type == InferenceType.IMAGE_STORY:
            return {
                "id": idx,
                "images": [self._load_image(row["processed_image_path"])],
                "stories": [row["story_structure"]],
                "questions": [row["question"]],
                "metadata": metadata
            }
        
        elif self.inference_type == InferenceType.STORY_ONLY:
            return {
                "id": idx,
                "stories": [row["story_structure"]],
                "questions": [row["question"]],
                "metadata": metadata
            }
        
        elif self.inference_type == InferenceType.IMAGE_CAPTIONS:
            return {
                "id": idx,
                "images": [self._load_image(row["processed_image_path"])],
                "captions": [captions],
                "questions": [row["question"]],
                "metadata": metadata
            }
        
        elif self.inference_type == InferenceType.CAPTIONS_ONLY:
            return {
                "id": idx,
                "captions": [captions],
                "questions": [row["question"]],
                "metadata": metadata
            }
        
        elif self.inference_type == InferenceType.STORY_CAPTIONS:
            return {
                "id": idx,
                "stories": [row["story_structure"]],
                "captions": [captions],
                "questions": [row["question"]],
                "metadata": metadata
            }
        
        elif self.inference_type == InferenceType.STORY_DETAILS:
            return {
                "id": idx,
                "images": [self._load_image(row["processed_image_path"])],
                "stories": [row["story_details"]],
                "questions": [row["question"]],
                "metadata": metadata
            }
        
        else:
            raise ValueError(f"Unsupported inference type: {self.inference_type}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def get_original_data(self) -> pd.DataFrame:
        """Return the original dataframe for result saving."""
        return self.data 