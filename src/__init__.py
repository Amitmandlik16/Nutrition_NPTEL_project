"""
Nutrition Label Extraction System
==================================
A Vision-Language Model based system for extracting nutrition information from food labels.

Main Modules:
- image_processor: Image quality assessment and preprocessing
- vlm_extractor: Gemini Vision model based extraction
- schema_validator: JSON schema validation and numeric validation
- evaluation: Evaluation metrics and comparison with baseline
- utils: Utility functions and helpers
"""

"""
Nutrition Label Extraction System
==================================
Package initialization.
"""

__version__ = "1.0.0"

from .image_processor import ImageQualityAssessor, ImagePreprocessor
from .vlm_extractor import NutritionVLMExtractor
from .schema_validator import NutritionSchemaValidator
from .evaluation import NutritionEvaluator

__all__ = [
    "ImageQualityAssessor",
    "ImagePreprocessor",
    "NutritionVLMExtractor",
    "NutritionSchemaValidator",
    "NutritionEvaluator",
]