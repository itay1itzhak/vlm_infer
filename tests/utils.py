from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

def create_test_image(size=(224, 224)):
    """Create a test image."""
    return Image.fromarray(
        np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    )

def create_test_dataset(num_samples=5):
    """Create a test VToM dataset."""
    data = {
        "story_structure": [f"Story {i}" for i in range(num_samples)],
        "question": [f"Question {i}?" for i in range(num_samples)],
        "expected_answer": [f"Answer {i}" for i in range(num_samples)],
        "combined_image_path": [f"path/to/image_{i}.png" for i in range(num_samples)],
        "story_type": ["type_A", "type_B"] * (num_samples // 2 + num_samples % 2),
        "num_people": [2, 3, 2, 4, 3][:num_samples]
    }
    return pd.DataFrame(data)

def create_test_results(num_samples=5, models=None):
    """Create test results dataframe."""
    models = models or ["model_generation", "model_generation"]
    df = create_test_dataset(num_samples)
    
    for model in models:
        df[model] = [f"Generated answer {i} from {model}" for i in range(num_samples)]
        df[f"{model}_processed"] = [f"Answer {i}" for i in range(num_samples)]
        df[f"{model}_accuracy"] = np.random.uniform(0, 1, num_samples)
    
    return df 