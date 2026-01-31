"""
Main Script - Example Nutrition Label Extraction Pipeline
==========================================================
Complete working example showing all system components.

This script demonstrates:
1. Image quality assessment
2. VLM-based extraction
3. Validation and normalization
4. Evaluation metrics
5. Results saving

Usage:
    python main.py <image_path>
    python main.py nutrition_label.jpg
"""

import sys
import json
import logging
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Any

# Import modules
from src.image_processor import ImageQualityAssessor, ImagePreprocessor
from src.vlm_extractor import NutritionVLMExtractor
from src.schema_validator import NutritionSchemaValidator
from src.evaluation import NutritionEvaluator
from src import utils
import config

# Setup logging
logger = logging.getLogger(__name__)
utils.setup_logging(level=config.LOG_LEVEL)


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def assess_image_quality(image_path: str) -> bool:
    """
    Step 1: Assess image quality.
    
    Returns:
        True if quality is acceptable, False otherwise
    """
    print_section("STEP 1: Image Quality Assessment")
    
    try:
        assessor = ImageQualityAssessor()
        quality_score = assessor.assess(image_path)
        
        print(f"Overall Score:        {quality_score.overall_score:.3f}")
        print(f"  Brightness:         {quality_score.brightness:.3f}")
        print(f"  Contrast:           {quality_score.contrast:.3f}")
        print(f"  Sharpness:          {quality_score.blur:.3f}")
        print(f"  Resolution:         {quality_score.resolution:.3f}")
        
        if quality_score.issues:
            print(f"\n⚠️  Quality Issues Detected:")
            for issue in quality_score.issues:
                print(f"   - {issue}")
        else:
            print(f"\n✓ Image quality is good!")
        
        return quality_score.is_valid
    
    except Exception as e:
        logger.error(f"Error assessing image quality: {e}")
        return False


def preprocess_image(image_path: str) -> str:
    """
    Step 2: Preprocess image (optional).
    
    Returns:
        Path to preprocessed image
    """
    print_section("STEP 2: Image Preprocessing (Optional)")
    
    try:
        preprocessor = ImagePreprocessor()
        processed_img, output_path = preprocessor.preprocess(image_path)
        
        print(f"✓ Preprocessing complete")
        print(f"  Output saved to: {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image_path  # Return original if preprocessing fails


def extract_nutrition(image_path: str) -> Dict[str, Any]:
    """
    Step 3: Extract nutrition using VLM.
    
    Returns:
        Extracted nutrition data
    """
    print_section("STEP 3: VLM-Based Extraction")
    
    try:
        extractor = NutritionVLMExtractor()
        result = extractor.extract(image_path)
        
        print(f"✓ Extraction complete in {result.extraction_time:.2f} seconds")
        print(f"  Nutrients found: {len(result.nutrients)}")
        
        # Display ingredients if found
        if result.ingredients and result.ingredients.list:
            print(f"  Ingredients found: {len(result.ingredients.list)}")
            if result.ingredients.allergens:
                print(f"  Allergens identified: {len(result.ingredients.allergens)}")
        
        if result.nutrients:
            print(f"\n  Extracted nutrients:")
            for i, nutrient in enumerate(result.nutrients, 1):
                confidence = nutrient.confidence
                value_str = f"{nutrient.value} {nutrient.unit}" if nutrient.value else "N/A"
                print(f"    {i:2d}. {nutrient.name:20s} {value_str:20s} "
                      f"[{confidence:.1%} confidence]")
        
        # Display ingredients
        if result.ingredients and result.ingredients.list:
            print(f"\n  Ingredients list (in order):")
            for i, ingredient in enumerate(result.ingredients.list[:10], 1):  # Show first 10
                print(f"    {i:2d}. {ingredient}")
            if len(result.ingredients.list) > 10:
                print(f"    ... and {len(result.ingredients.list) - 10} more")
            
            if result.ingredients.allergens:
                print(f"\n  ⚠️  Allergens detected:")
                for allergen in result.ingredients.allergens:
                    print(f"    • {allergen}")
        
        return {
            "serving_size": result.serving_size,
            "servings_per_container": result.servings_per_container,
            "nutrients": [
                {
                    "name": n.name,
                    "value": n.value,
                    "unit": n.unit,
                    "per_rda": n.per_rda,
                    "confidence": n.confidence
                }
                for n in result.nutrients
            ],
            "ingredients": {
                "list": result.ingredients.list,
                "allergens": result.ingredients.allergens,
                "confidence": result.ingredients.confidence
            } if result.ingredients else None
        }
    
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise


def validate_extraction(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 4: Validate and normalize extracted data.
    
    Returns:
        Validated and normalized data
    """
    print_section("STEP 4: Validation & Normalization")
    
    try:
        validator = NutritionSchemaValidator()
        report = validator.validate(extracted_data)
        
        if report.is_valid:
            print(f"✓ Validation passed!")
        else:
            print(f"⚠️  Validation issues found:")
            
            if report.errors:
                print(f"\n  Errors:")
                for error in report.errors:
                    print(f"    - {error.field}: {error.error}")
            
            if report.warnings:
                print(f"\n  Warnings:")
                for warning in report.warnings:
                    print(f"    - {warning}")
        
        # Show normalized data
        normalized = report.normalized_data
        nutrients = normalized.get("nutrients", [])
        
        print(f"\n  After filtering (confidence >= {config.MIN_CONFIDENCE_THRESHOLD}):")
        print(f"    Nutrients retained: {len(nutrients)}")
        
        return normalized
    
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        raise


def save_results(extracted_data: Dict[str, Any], image_path: str) -> str:
    """
    Step 5: Save extracted results.
    
    Returns:
        Path to saved results file
    """
    print_section("STEP 5: Saving Results")
    
    try:
        # Create results directory
        results_dir = Path(config.RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        image_name = Path(image_path).stem
        timestamp = utils.get_timestamp_string()
        output_file = results_dir / f"{image_name}_{timestamp}.json"
        
        # Save as JSON
        utils.save_json(extracted_data, str(output_file))
        
        print(f"✓ Results saved to: {output_file}")
        print(f"  File size: {utils.get_file_size_mb(str(output_file)):.2f} MB")
        
        return str(output_file)
    
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


def print_summary(extracted_data: Dict[str, Any]):
    """Print final summary."""
    print_section("SUMMARY")
    
    serving = extracted_data.get("serving_size", {})
    nutrients = extracted_data.get("nutrients", [])
    ingredients = extracted_data.get("ingredients")
    
    print(f"Serving Size:       {serving.get('quantity', 'N/A')} {serving.get('unit', '')}")
    print(f"Nutrients Extracted: {len(nutrients)}")
    
    if ingredients:
        ingredients_list = ingredients.get("list", [])
        allergens = ingredients.get("allergens", [])
        print(f"Ingredients Found:   {len(ingredients_list)}")
        if allergens:
            print(f"Allergens Detected:  {len(allergens)} ({', '.join(allergens)})")
    
    if nutrients:
        # Calculate statistics
        confidences = [n.get("confidence", 0) for n in nutrients]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        print(f"Average Confidence: {avg_confidence:.1%}")
        
        # Show nutrient summary
        print(f"\nNutrient Summary:")
        for nutrient in nutrients[:5]:  # Show first 5
            value_str = f"{nutrient.get('value', 'N/A')} {nutrient.get('unit', '')}"
            print(f"  • {nutrient['name']:20s}: {value_str}")
        
        if len(nutrients) > 5:
            print(f"  ... and {len(nutrients) - 5} more")
    
    print(f"\n✓ Pipeline execution successful!")


def main():
    """Main execution function."""
    # Parse arguments
    parser = ArgumentParser(description="Nutrition Label Extraction Pipeline")
    parser.add_argument("image", help="Path to nutrition label image")
    parser.add_argument("--skip-quality-check", action="store_true",
                       help="Skip image quality assessment")
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Skip image preprocessing")
    parser.add_argument("--output", help="Output file path (optional)")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.image).exists():
        print(f"❌ Error: File not found: {args.image}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("  🍎 NUTRITION LABEL EXTRACTION SYSTEM")
    print(f"  Processing: {args.image}")
    print("="*60)
    
    try:
        # Step 1: Quality assessment
        if not args.skip_quality_check:
            if not assess_image_quality(args.image):
                print("\n⚠️  Image quality is below threshold.")
                print("    Continue? (y/n): ", end="")
                if input().lower() != 'y':
                    print("❌ Aborted by user")
                    sys.exit(1)
        
        # Step 2: Preprocessing
        image_to_process = args.image
        if not args.skip_preprocessing:
            image_to_process = preprocess_image(args.image)
        
        # Step 3: Extract
        extracted_data = extract_nutrition(image_to_process)
        
        # Step 4: Validate
        validated_data = validate_extraction(extracted_data)
        
        # Step 5: Save
        output_path = save_results(
            validated_data,
            args.output or args.image
        )
        
        # Summary
        print_summary(validated_data)
        
        print(f"\n📄 Results saved to: {output_path}\n")
    
    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logger.exception("Pipeline error")
        sys.exit(1)


if __name__ == "__main__":
    main()
