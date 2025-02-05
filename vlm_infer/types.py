from enum import Enum
from typing import Union, List, Dict, Optional, Any
from dataclasses import dataclass
from PIL import Image

class InferenceType(Enum):
    """Types of inference supported by the system."""
    STANDARD = "standard"  # image + question
    IMAGE_STORY = "image_story"  # image + story + question
    STORY_ONLY = "story_only"  # story + question
    IMAGE_CAPTIONS = "image_captions"  # image + captions + question
    CAPTIONS_ONLY = "captions_only"  # captions + question
    STORY_CAPTIONS = "story_captions"  # story + captions + question
    STORY_DETAILS = "story_details"  # image + story_details + question

@dataclass
class InferenceInput:
    question: str
    images: Optional[List[Image.Image]] = None
    story: Optional[str] = None
    captions: Optional[Union[str, List[str], Dict[str, str]]] = None
    details: Optional[Dict[str, Any]] = None
    inference_type: InferenceType = InferenceType.STANDARD
    
    def validate(self):
        """Validate input combination based on inference type."""
        if self.inference_type == InferenceType.STANDARD:
            assert self.images is not None and len(self.images) > 0
            assert self.story is None and self.captions is None
            
        elif self.inference_type == InferenceType.IMAGE_STORY:
            assert self.images is not None and len(self.images) > 0
            assert self.story is not None
            assert self.captions is None
            
        elif self.inference_type == InferenceType.STORY_ONLY:
            assert self.images is None
            assert self.story is not None
            assert self.captions is None
            
        elif self.inference_type == InferenceType.IMAGE_CAPTIONS:
            assert self.images is not None and len(self.images) > 0
            assert self.captions is not None
            assert self.story is None
            
        elif self.inference_type == InferenceType.CAPTIONS_ONLY:
            assert self.images is None
            assert self.captions is not None
            assert self.story is None
            
        elif self.inference_type == InferenceType.STORY_CAPTIONS:
            assert self.images is None
            assert self.story is not None
            assert self.captions is not None
            
        elif self.inference_type == InferenceType.STORY_DETAILS:
            assert self.images is not None and len(self.images) > 0
            assert self.story is not None
            assert self.details is not None
            assert self.captions is None 