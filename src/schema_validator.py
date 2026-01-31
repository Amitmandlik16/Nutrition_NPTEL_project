"""
Schema Validation Module
========================
Validates extracted data against JSON schema and applies business rules.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from jsonschema import validate, ValidationError as JsonSchemaError # <-- FIX: Alias the import
import config

logger = logging.getLogger(__name__)

# --- FIX: Renamed class to avoid conflict ---
@dataclass
class ValidationIssue:
    """Container for validation errors."""
    field: str
    value: Any
    error: str
    severity: str  # "error", "warning", "info"

@dataclass
class ValidationReport:
    """Complete validation report."""
    is_valid: bool
    errors: List[ValidationIssue] # <-- Updated reference
    warnings: List[str]
    normalized_data: Dict[str, Any]

class NutrientValidator:
    def __init__(self):
        self.unit_conversions = config.UNIT_CONVERSION
        self.rda_values = config.NUTRIENT_RDA
        self.numeric_tolerance = config.NUMERIC_TOLERANCE

    def normalize_unit(self, unit: str) -> str:
        if not unit: return None
        unit = unit.lower().strip().rstrip('s') # remove plurals
        return config.ACCEPTED_UNITS.get(unit, unit)

class NutritionSchemaValidator:
    def __init__(self):
        self.schema_path = "data/nutrition_schema.json"
        self.nutrient_validator = NutrientValidator()
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict:
        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            return {}

    def _validate_schema(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        errors = []
        if not self.schema: return errors
        
        try:
            # --- FIX: Use the aliased exception name ---
            validate(instance=data, schema=self.schema)
        except JsonSchemaError as e:
            # Extract clean path (e.g., "nutrients.0.value")
            path = ".".join([str(x) for x in e.path]) if e.path else "root"
            errors.append(ValidationIssue(
                field=path,
                value=None,
                error=e.message,
                severity="error"
            ))
        return errors

    def validate(self, data: Dict[str, Any]) -> ValidationReport:
        # 1. Structural Validation
        schema_errors = self._validate_schema(data)
        
        # 2. Logical Validation & Normalization
        warnings = []
        normalized = data.copy()
        
        # Example rule: Check confidence
        if "nutrients" in data:
            valid_nutrients = []
            for n in data["nutrients"]:
                if n.get("confidence", 0) < 0.5:
                    warnings.append(f"Low confidence for {n.get('name')}")
                else:
                    valid_nutrients.append(n)
            normalized["nutrients"] = valid_nutrients

        return ValidationReport(
            is_valid=len(schema_errors) == 0,
            errors=schema_errors,
            warnings=warnings,
            normalized_data=normalized
        )