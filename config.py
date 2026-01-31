"""
Configuration Module
====================
Central configuration file for the Nutrition Label Extraction System.
Contains all tunable parameters, API settings, and model configurations.

This module is designed for easy modification without touching core logic.
Change values here to adjust system behavior across all modules.
"""

import os
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API & MODEL CONFIGURATION
# ============================================================================

# Gemini Vision API Key (get from Google Cloud Console)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyC2SPG9YXxjszQbsyAV2YF9f2p7ZMGjMzc")

# Vision-Language Model to use
# Best options: "gemini-2.5-flash" (fast), "gemini-2.5-pro" (accurate), "gemini-2.0-flash" (newer)
VLM_MODEL = "gemini-2.5-flash"

# Temperature for VLM responses (0.0 = deterministic, 1.0 = creative)
# Lower values = more consistent, Higher = more varied
VLM_TEMPERATURE = 0.2

# Maximum tokens for VLM response (increased to handle complete nutrition labels)
VLM_MAX_TOKENS = 4096

# ============================================================================
# IMAGE PROCESSING CONFIGURATION
# ============================================================================

# Image quality assessment thresholds (lowered to accept more images)
IMAGE_QUALITY = {
    "min_width": 300,              # Minimum acceptable image width (lowered)
    "min_height": 300,             # Minimum acceptable image height (lowered)
    "min_brightness": 20,          # Minimum average brightness (0-255, lowered)
    "max_brightness": 240,         # Maximum average brightness (0-255, increased)
    "min_contrast": 5,             # Minimum contrast threshold (lowered for low-contrast images)
    "blur_threshold": 50.0,        # Threshold for blur detection (lowered to accept blurry images)
}

# Image preprocessing settings (enhanced for low-quality images)
PREPROCESSING = {
    "resize_max_width": 1920,      # Maximum width after resize
    "resize_max_height": 1080,     # Maximum height after resize
    "target_dpi": 300,             # Target DPI for optimal OCR
    "apply_clahe": True,           # Apply CLAHE for contrast enhancement
    "clahe_clip_limit": 3.0,       # CLAHE clip limit (increased for stronger contrast)
    "clahe_tile_size": 8,          # CLAHE tile size
    "enable_denoising": True,      # Enable bilateral filter denoising
}

# ============================================================================
# JSON SCHEMA & VALIDATION
# ============================================================================

# Accepted nutrient units (can be extended)
ACCEPTED_UNITS = {
    "g": "gram",
    "mg": "milligram",
    "μg": "microgram",
    "mcg": "microgram",
    "kcal": "kilocalorie",
    "cal": "calorie",
    "%": "percentage",
    "mg/ml": "milligram per milliliter",
    "g/100ml": "gram per 100 milliliter",
}

# Nutrient categories with daily reference values (RDA)
NUTRIENT_RDA = {
    "energy": {"unit": "kcal", "rda": 2000},
    "protein": {"unit": "g", "rda": 50},
    "total_fat": {"unit": "g", "rda": 78},
    "saturated_fat": {"unit": "g", "rda": 20},
    "trans_fat": {"unit": "g", "rda": 0},
    "cholesterol": {"unit": "mg", "rda": 300},
    "sodium": {"unit": "mg", "rda": 2300},
    "total_carbohydrate": {"unit": "g", "rda": 275},
    "dietary_fiber": {"unit": "g", "rda": 28},
    "sugars": {"unit": "g", "rda": 50},
    "calcium": {"unit": "mg", "rda": 1300},
    "iron": {"unit": "mg", "rda": 18},
    "potassium": {"unit": "mg", "rda": 4700},
    "vitamin_d": {"unit": "μg", "rda": 20},
    "vitamin_c": {"unit": "mg", "rda": 90},
}

# ============================================================================
# EXTRACTION PROMPTS
# ============================================================================

# System prompt for Gemini Vision model
SYSTEM_PROMPT = """You are an expert nutrition label analyzer. Your task is to extract 
nutrition information from food package labels with high accuracy.

IMPORTANT RULES:
1. Extract ONLY visible information from the label
2. Be precise with numeric values (no approximations)
3. Capture exact units as shown on label
4. If %RDA is shown, extract it; if not, set to null
5. Return ONLY valid JSON, no extra text
6. For each nutrient, provide a confidence score (0-1)"""

# User prompt template for image extraction
# config.py

EXTRACTION_PROMPT_TEMPLATE = """Extract all nutrition information and ingredients from this food label.
Return ONLY a valid JSON object with structure:
{{
    "serving_size": {{"quantity": number, "unit": "string"}},
    "servings_per_container": number,
    "nutrients": [
        {{
            "name": "string",
            "value": number,
            "unit": "string",
            "per_rda": number,
            "confidence": 0-1
        }}
    ],
    "ingredients": {{
        "list": ["ingredient1", "ingredient2", ...],
        "allergens": ["allergen1", "allergen2", ...],
        "confidence": 0-1
    }}
}}

IMPORTANT RULES:
1. **NUTRIENT VALUES**: Extract numeric mass/energy values from the "Per 100g" or "Per 100ml" column if available.
2. **RDA / %DV**: Extract the percentage from the "% Daily Value" or "% DV" column. 
   - **CRITICAL:** The %DV column is often next to the "Per Serving" column. You MUST extract the %DV even if you are extracting the nutrient mass from the 100g column.
   - Extract ONLY the number (e.g., if label says "15%", return 15).
3. If no "100g" column exists, use the "Per Serving" column for everything.
4. Extract ingredients in order.
5. Return null for missing fields.
"""

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Tolerance for numeric accuracy evaluation
NUMERIC_TOLERANCE = {
    "energy": 5,           # kcal tolerance
    "macros": 0.5,         # grams tolerance for protein, fat, carbs
    "micros": 2,           # mg/μg tolerance for vitamins/minerals
    "percent_rda": 3,      # % tolerance for RDA values
}

# Weights for evaluation metrics (sum to 1.0)
EVALUATION_WEIGHTS = {
    "field_accuracy": 0.3,     # Precision of extracted fields
    "numeric_accuracy": 0.4,   # Accuracy of numeric values
    "completeness": 0.2,       # Percentage of nutrients extracted
    "confidence": 0.1,         # Model confidence scores
}

# ============================================================================
# BASELINE OCR CONFIGURATION (Optional)
# ============================================================================

# Tesseract path (for Windows users)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# OCR configuration
OCR_CONFIG = {
    "lang": "eng",
    "config": "--psm 6",  # PSM mode (6 = assume single text block)
}

# ============================================================================
# STREAMLIT UI CONFIGURATION
# ============================================================================

STREAMLIT_CONFIG = {
    "page_title": "Nutrition Label Extractor",
    "page_icon": "🍎",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_upload_size": 10,  # MB
}

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

# Logging level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_LEVEL = "INFO"

# Enable detailed logging for debugging
DEBUG_MODE = True

# Save extracted results to disk
SAVE_RESULTS = True
RESULTS_DIR = "./results"

# ============================================================================
# UNIT NORMALIZATION RULES
# ============================================================================

# Unit conversion factors (to base units: g, mg, μg, kcal)
UNIT_CONVERSION = {
    # Weight conversions to grams
    "kg": 1000,
    "g": 1,
    "mg": 0.001,
    "μg": 0.000001,
    "mcg": 0.000001,
    
    # Energy conversions to kcal
    "kcal": 1,
    "cal": 0.001,
    "kJ": 0.239,  # kilojoules to kcal
    
    # Volume conversions
    "ml": 1,
    "l": 1000,
    "fl oz": 29.5735,
}

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

# Minimum confidence to include extracted nutrient
MIN_CONFIDENCE_THRESHOLD = 0.7

# Confidence levels for different extraction types
CONFIDENCE_LEVELS = {
    "high": 0.85,      # Very confident
    "medium": 0.70,    # Reasonably confident
    "low": 0.50,       # Some uncertainty
    "reject": 0.0,     # Do not use
}

# ============================================================================
# FILE PATHS
# ============================================================================

DATA_DIR = "./data"
SCHEMA_FILE = "./data/nutrition_schema.json"
RESULTS_DIR = "./results"
EXAMPLES_DIR = "./examples"
MODELS_DIR = "./models"  # For storing trained validation models if needed
