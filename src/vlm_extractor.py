"""
Vision-Language Model Extraction Module
========================================
Handles extraction of nutrition information using Gemini Vision API.

Design Decision:
- VLMs directly process images → preserves visual structure and layout
- Avoids OCR noise from separate text recognition step
- Structured prompt ensures JSON output format
- Built-in capability to understand table structures and relationships

Why VLM > OCR+LLM:
1. Visual context: Understands nutrition label layouts
2. Fewer errors: No intermediate OCR mistakes
3. Units clarity: Can distinguish units by position/context
4. Better accuracy: 15-25% improvement typical

Key Classes:
- NutritionVLMExtractor: Main extraction engine
"""
"""
Vision-Language Model Extraction Module
========================================
Handles extraction of nutrition information using Gemini Vision API.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import time

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

import config

logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """Tracks API token usage for cost calculation."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

@dataclass
class ExtractedNutrient:
    """Container for a single extracted nutrient."""
    name: str
    value: Optional[float]
    unit: Optional[str]
    per_rda: Optional[float]
    confidence: float

@dataclass
class ExtractedIngredients:
    """Container for ingredients extraction."""
    list: List[str]
    allergens: List[str]
    confidence: float

@dataclass
class ExtractedNutritionLabel:
    """Container for complete extraction results."""
    serving_size: Optional[Dict[str, Any]] = None
    servings_per_container: Optional[float] = None
    nutrients: List[ExtractedNutrient] = field(default_factory=list)
    ingredients: Optional[ExtractedIngredients] = None
    extraction_time: float = 0.0
    raw_response: str = ""
    api_calls_made: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)

class NutritionVLMExtractor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in config or environment")
        
        genai.configure(api_key=self.api_key)
        
        # --- FIX 1: Ensure model_name is defined here ---
        self.model_name = config.VLM_MODEL
        
        self.generation_config = {
            "temperature": config.VLM_TEMPERATURE,
            "max_output_tokens": config.VLM_MAX_TOKENS,
            # Removed response_mime_type for compatibility with older library versions
        }
        
        logger.info(f"Initialized Gemini Vision extractor with model: {self.model_name}")

    def _encode_image(self, image_path: str) -> list:
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        with open(img_path, "rb") as f:
            image_data = f.read()
            
        mime_type = "image/jpeg"
        if img_path.suffix.lower() == ".png":
            mime_type = "image/png"
            
        return [{"mime_type": mime_type, "data": image_data}]

    def _call_vlm(self, image_parts: list, prompt: str) -> Tuple[str, TokenUsage]:
        """Call Gemini API and return response text AND token usage."""
        model = genai.GenerativeModel(self.model_name)
        
        try:
            response = model.generate_content(
                [prompt, *image_parts],
                generation_config=self.generation_config
            )
            
            # Capture Token Usage
            usage = TokenUsage()
            
            # The library update (Step 1) makes this work:
            if hasattr(response, 'usage_metadata'):
                usage.input_tokens = response.usage_metadata.prompt_token_count
                usage.output_tokens = response.usage_metadata.candidates_token_count
                usage.total_tokens = response.usage_metadata.total_token_count
            
            return response.text, usage
            
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            raise

    def _parse_response(self, json_text: str) -> Dict[str, Any]:
        try:
            clean_text = json_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error: {e}")
            return {}

    def extract(self, image_path: str) -> ExtractedNutritionLabel:
        """Main extraction method."""
        start_time = time.time()
        image_parts = self._encode_image(image_path)
        prompt = config.EXTRACTION_PROMPT_TEMPLATE
        
        try:
            response_text, token_usage = self._call_vlm(image_parts, prompt)
            raw_data = self._parse_response(response_text)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ExtractedNutritionLabel()

        # Parse Nutrients
        nutrients = []
        for n in raw_data.get("nutrients", []):
            nutrients.append(ExtractedNutrient(
                name=n.get("name", "Unknown"),
                value=n.get("value"),
                unit=n.get("unit"),
                per_rda=n.get("per_rda"),
                confidence=n.get("confidence", 0.0)
            ))
            
        # Parse Ingredients
        ingredients = None
        if raw_data.get("ingredients"):
            ing_data = raw_data["ingredients"]
            ingredients = ExtractedIngredients(
                list=ing_data.get("list", []),
                allergens=ing_data.get("allergens", []),
                confidence=ing_data.get("confidence", 0.0)
            )

        return ExtractedNutritionLabel(
            serving_size=raw_data.get("serving_size"),
            servings_per_container=raw_data.get("servings_per_container"),
            nutrients=nutrients,
            ingredients=ingredients,
            extraction_time=time.time() - start_time,
            raw_response=response_text,
            api_calls_made=1,
            token_usage=token_usage
        )