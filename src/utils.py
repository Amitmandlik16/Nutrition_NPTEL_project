"""
Utility Functions Module
========================
Helper functions for common tasks across the pipeline.

Contains:
- JSON handling utilities
- Image utilities
- File operations
- Logging helpers
- Data formatting functions
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

import config

logger = logging.getLogger(__name__)


# ============================================================================
# JSON UTILITIES
# ============================================================================

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise


def save_json(data: Dict[str, Any], file_path: str, 
              pretty: bool = True) -> str:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        pretty: Whether to pretty-print JSON
        
    Returns:
        Path to saved file
    """
    try:
        # Create parent directory if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        logger.debug(f"Saved JSON to {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        raise


def dict_to_json_string(data: Dict[str, Any], pretty: bool = True) -> str:
    """Convert dictionary to JSON string."""
    if pretty:
        return json.dumps(data, indent=2, ensure_ascii=False)
    else:
        return json.dumps(data, ensure_ascii=False)


def json_string_to_dict(json_str: str) -> Dict[str, Any]:
    """Parse JSON string to dictionary."""
    return json.loads(json_str)


# ============================================================================
# FILE UTILITIES
# ============================================================================

def ensure_directory(dir_path: str) -> str:
    """Create directory if it doesn't exist."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def list_files_by_extension(directory: str, extension: str) -> List[str]:
    """
    List all files with given extension in directory.
    
    Args:
        directory: Directory to search
        extension: File extension (e.g., '.jpg', '.txt')
        
    Returns:
        List of file paths
    """
    extension = extension if extension.startswith('.') else f".{extension}"
    return [
        str(f) for f in Path(directory).rglob(f"*{extension}")
    ]


def get_timestamp_string(format: str = "%Y%m%d_%H%M%S") -> str:
    """Get current timestamp as formatted string."""
    return datetime.now().strftime(format)


def backup_file(file_path: str) -> str:
    """
    Create backup of file with timestamp.
    
    Returns:
        Path to backup file
    """
    file_path = Path(file_path)
    timestamp = get_timestamp_string()
    backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    
    import shutil
    shutil.copy(file_path, backup_path)
    logger.debug(f"Created backup: {backup_path}")
    
    return str(backup_path)


# ============================================================================
# TEXT PROCESSING UTILITIES
# ============================================================================

def normalize_unit(unit_str: str) -> Optional[str]:
    """
    Normalize unit string to standard form.
    
    Examples:
        "mgm" -> "mg"
        "Grams" -> "g"
        "kcal" -> "kcal"
    """
    if not unit_str:
        return None
    
    # Remove spaces
    unit_str = unit_str.strip().lower()
    
    # Common replacements
    replacements = {
        "grams": "g",
        "gm": "g",
        "mg": "mg",
        "milligrams": "mg",
        "microgram": "μg",
        "mcg": "μg",
        "calories": "kcal",
        "kilocalories": "kcal",
        "percent": "%",
        "rda": "%",
    }
    
    for old, new in replacements.items():
        if old in unit_str:
            return new
    
    return unit_str if unit_str in config.ACCEPTED_UNITS else None


def extract_number(text: str) -> Optional[float]:
    """
    Extract numeric value from text.
    
    Examples:
        "5.5 grams" -> 5.5
        "100mg" -> 100
        "2,500 kcal" -> 2500
    """
    if not text:
        return None
    
    # Remove currency symbols and common text
    text = text.replace("$", "").replace(",", "")
    
    # Find number with optional decimal
    match = re.search(r'(\d+\.?\d*)', str(text))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    
    return None


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that aren't needed
    text = re.sub(r'[^\w\s\-./()%]', '', text)
    
    return text.strip()


# ============================================================================
# DATA FORMATTING UTILITIES
# ============================================================================

def format_nutrient_for_display(nutrient: Dict[str, Any]) -> str:
    """Format nutrient for human-readable display."""
    name = nutrient.get("name", "Unknown")
    value = nutrient.get("value")
    unit = nutrient.get("unit", "")
    confidence = nutrient.get("confidence", 0)
    
    value_str = f"{value} {unit}".strip() if value is not None else "N/A"
    
    return f"{name}: {value_str} (confidence: {confidence:.1%})"


def format_metrics_for_display(metrics: Dict[str, Any]) -> str:
    """Format evaluation metrics for display."""
    lines = []
    lines.append("Evaluation Metrics:")
    lines.append(f"  Overall Score: {metrics.get('overall_score', 0):.3f}")
    lines.append(f"  Field Accuracy: {metrics.get('field_accuracy', 0):.1%}")
    lines.append(f"  Numeric Accuracy: {metrics.get('numeric_accuracy', 0):.1%}")
    lines.append(f"  Completeness: {metrics.get('completeness', 0):.1%}")
    lines.append(f"  Average Confidence: {metrics.get('confidence_score', 0):.2f}")
    
    return "\n".join(lines)


def format_error_for_display(error: Dict[str, Any]) -> str:
    """Format validation error for display."""
    return f"[{error.get('severity', 'ERROR')}] {error.get('field', 'Unknown')}: {error.get('error', 'Unknown error')}"


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_logging(log_file: Optional[str] = None, 
                  level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional file to log to
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup handlers
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        ensure_directory(str(Path(log_file).parent))
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def is_valid_nutrient_name(name: str) -> bool:
    """Check if string is a valid nutrient name."""
    if not name or not isinstance(name, str):
        return False
    
    # Must be at least 3 characters and contain letters
    return len(name.strip()) >= 3 and any(c.isalpha() for c in name)


def is_valid_numeric_value(value: Any) -> bool:
    """Check if value is valid number."""
    if value is None:
        return True
    
    try:
        num = float(value)
        return -10000 < num < 1000000  # Reasonable range for nutrients
    except (ValueError, TypeError):
        return False


def is_valid_confidence_score(confidence: float) -> bool:
    """Check if confidence score is valid (0-1)."""
    try:
        conf = float(confidence)
        return 0 <= conf <= 1
    except (ValueError, TypeError):
        return False


# ============================================================================
# BATCH PROCESSING UTILITIES
# ============================================================================

def batch_process(items: List[Any], batch_size: int, 
                 process_func) -> List[Any]:
    """
    Process items in batches.
    
    Args:
        items: Items to process
        batch_size: Size of each batch
        process_func: Function to apply to each batch
        
    Returns:
        List of results
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} items)")
        
        try:
            batch_result = process_func(batch)
            results.extend(batch_result)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
    
    return results


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from batch processing.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Aggregated statistics
    """
    if not results:
        return {}
    
    aggregated = {
        "total": len(results),
        "successful": len([r for r in results if r and r.get("success", True)]),
        "errors": len([r for r in results if r and not r.get("success", True)]),
    }
    
    return aggregated


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def setup_project_directories() -> Dict[str, str]:
    """
    Create all required project directories.
    
    Returns:
        Dictionary of directory paths
    """
    dirs = {
        "data": config.DATA_DIR,
        "results": config.RESULTS_DIR,
        "examples": config.EXAMPLES_DIR,
        "models": config.MODELS_DIR,
    }
    
    for dir_path in dirs.values():
        ensure_directory(dir_path)
    
    logger.info("Project directories created/verified")
    return dirs


def print_config_summary():
    """Print summary of current configuration."""
    print("\n" + "=" * 60)
    print("NUTRITION EXTRACTION SYSTEM - CONFIGURATION")
    print("=" * 60)
    print(f"VLM Model: {config.VLM_MODEL}")
    print(f"Min Confidence: {config.MIN_CONFIDENCE_THRESHOLD}")
    print(f"Debug Mode: {config.DEBUG_MODE}")
    print(f"Save Results: {config.SAVE_RESULTS}")
    print("=" * 60 + "\n")
