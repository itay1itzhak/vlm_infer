from typing import Dict, Any, List, Optional, Union
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class BaseAnswerExtractor:
    """Base class for answer extraction strategies."""
    def extract(self, text: str) -> str:
        raise NotImplementedError

class RegexAnswerExtractor(BaseAnswerExtractor):
    """Extract answers using regex patterns."""
    def __init__(self, patterns: List[str]):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def extract(self, text: str) -> str:
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return text.strip()

class LMAnswerExtractor(BaseAnswerExtractor):
    """Extract answers using a language model."""
    def __init__(self, model_name: str = "allenai/OLMo-7B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def extract(self, text: str) -> str:
        prompt = f"Extract the direct answer from this text: {text}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

class AnswerEvaluator:
    """Evaluate model answers against ground truth."""
    def __init__(
        self,
        extraction_method: str = "regex",
        patterns: List[str] = None,
        lm_model: str = "allenai/OLMo-7B"
    ):
        if extraction_method == "regex":
            default_patterns = [
                r"(?:the answer is|i think|therefore|so,?|in conclusion,?)?\s*[:\-]?\s*(.*)",
                r"^(.*?)(?:\.|$)"
            ]
            self.extractor = RegexAnswerExtractor(patterns or default_patterns)
        elif extraction_method == "lm":
            self.extractor = LMAnswerExtractor(lm_model)
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}")
    
    def process_answers(self, df: pd.DataFrame, model_column: str) -> pd.DataFrame:
        """Process model outputs to extract answers."""
        df = df.copy()
        df[f"{model_column}_processed"] = df[model_column].apply(
            self.extractor.extract
        )
        return df 