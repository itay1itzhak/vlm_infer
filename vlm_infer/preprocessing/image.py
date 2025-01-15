from typing import Union, List, Tuple, Optional
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np

class ImagePreprocessor:
    """Handles various image preprocessing operations."""
    
    @staticmethod
    def resize_image(
        image: Image.Image,
        size: Union[int, Tuple[int, int]],
        method: str = "bilinear"
    ) -> Image.Image:
        """Resize image to target size."""
        if isinstance(size, int):
            size = (size, size)
        
        resize_methods = {
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
            "nearest": Image.NEAREST
        }
        return image.resize(size, resize_methods.get(method, Image.BILINEAR))
    
    @staticmethod
    def center_crop(
        image: Image.Image,
        size: Union[int, Tuple[int, int]]
    ) -> Image.Image:
        """Center crop the image."""
        if isinstance(size, int):
            size = (size, size)
        return T.CenterCrop(size)(image)
    
    @staticmethod
    def normalize(
        image: Union[Image.Image, torch.Tensor],
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> torch.Tensor:
        """Normalize image tensor."""
        if isinstance(image, Image.Image):
            image = T.ToTensor()(image)
        return T.Normalize(mean=mean, std=std)(image)
    
    @staticmethod
    def pad_to_square(
        image: Image.Image,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """Pad image to square with specified background color."""
        w, h = image.size
        max_size = max(w, h)
        
        result = Image.new(image.mode, (max_size, max_size), background_color)
        result.paste(image, ((max_size - w) // 2, (max_size - h) // 2))
        return result
    
    @staticmethod
    def apply_color_jitter(
        image: Image.Image,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.1
    ) -> Image.Image:
        """Apply color jittering."""
        transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        return transform(image) 