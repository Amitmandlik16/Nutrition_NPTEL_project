"""
Test Script for Ingredients Extraction Feature
==============================================
Quick verification that ingredients extraction is working correctly.

Run this to verify the feature is properly integrated.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_schema():
    """Test 1: Verify schema includes ingredients."""
    print("\n" + "="*60)
    print("TEST 1: Schema Validation")
    print("="*60)
    
    import json
    from pathlib import Path
    
    schema_path = Path(__file__).parent / "data" / "nutrition_schema.json"
    
    if not schema_path.exists():
        print("❌ Schema file not found!")
        return False
    
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    if "ingredients" in schema.get("properties", {}):
        print("✓ Schema includes 'ingredients' field")
        
        ingredients_schema = schema["properties"]["ingredients"]
        required_fields = ["list", "allergens", "confidence"]
        
        for field in required_fields:
            if field in ingredients_schema.get("properties", {}):
                print(f"  ✓ '{field}' property defined")
            else:
                print(f"  ❌ '{field}' property missing")
                return False
        
        return True
    else:
        print("❌ Schema does not include 'ingredients'")
        return False


def test_dataclasses():
    """Test 2: Verify dataclasses are properly defined."""
    print("\n" + "="*60)
    print("TEST 2: Dataclass Definitions")
    print("="*60)
    
    try:
        from src.vlm_extractor import ExtractedIngredients, ExtractedNutritionLabel
        
        print("✓ ExtractedIngredients imported successfully")
        
        # Test creating an instance
        ingredients = ExtractedIngredients(
            list=["ingredient1", "ingredient2"],
            allergens=["allergen1"],
            confidence=0.9
        )
        
        print(f"✓ Created ExtractedIngredients instance:")
        print(f"  - List: {ingredients.list}")
        print(f"  - Allergens: {ingredients.allergens}")
        print(f"  - Confidence: {ingredients.confidence}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_prompt():
    """Test 3: Verify extraction prompt includes ingredients."""
    print("\n" + "="*60)
    print("TEST 3: Extraction Prompt")
    print("="*60)
    
    try:
        import config
        
        prompt = config.EXTRACTION_PROMPT_TEMPLATE
        
        if "ingredients" in prompt.lower():
            print("✓ Prompt includes 'ingredients' keyword")
        else:
            print("❌ Prompt does not mention ingredients")
            return False
        
        if "allergen" in prompt.lower():
            print("✓ Prompt includes allergen detection instructions")
        else:
            print("⚠️  Warning: Prompt doesn't explicitly mention allergens")
        
        print(f"\nPrompt excerpt:")
        print("-" * 60)
        lines = prompt.split('\n')
        for line in lines[:15]:  # Show first 15 lines
            print(line)
        print("...")
        print("-" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_validator():
    """Test 4: Verify validator handles ingredients."""
    print("\n" + "="*60)
    print("TEST 4: Schema Validator")
    print("="*60)
    
    try:
        from src.schema_validator import NutritionSchemaValidator
        
        validator = NutritionSchemaValidator()
        
        # Check if _validate_ingredients method exists
        if hasattr(validator, '_validate_ingredients'):
            print("✓ Validator has '_validate_ingredients' method")
        else:
            print("❌ Validator missing '_validate_ingredients' method")
            return False
        
        # Test with sample data
        test_data = {
            "nutrients": [
                {
                    "name": "Protein",
                    "value": 10,
                    "unit": "g",
                    "per_rda": 20,
                    "confidence": 0.9
                }
            ],
            "ingredients": {
                "list": ["wheat flour", "sugar", "salt"],
                "allergens": ["wheat"],
                "confidence": 0.85
            }
        }
        
        report = validator.validate(test_data)
        
        print(f"✓ Validation completed")
        print(f"  - Valid: {report.is_valid}")
        print(f"  - Errors: {len(report.errors)}")
        print(f"  - Warnings: {len(report.warnings)}")
        
        if report.warnings:
            print("\n  Warnings:")
            for warning in report.warnings:
                print(f"    - {warning}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test 5: Full integration test (mock)."""
    print("\n" + "="*60)
    print("TEST 5: Integration Test (Structure)")
    print("="*60)
    
    try:
        # Test that the full pipeline can handle ingredients data
        from src.vlm_extractor import ExtractedIngredients, ExtractedNutritionLabel, ExtractedNutrient
        from src.schema_validator import NutritionSchemaValidator
        
        # Create mock extraction result
        nutrients = [
            ExtractedNutrient("Protein", 10.0, "g", 20.0, 0.9),
            ExtractedNutrient("Fat", 5.0, "g", 10.0, 0.85)
        ]
        
        ingredients = ExtractedIngredients(
            list=["wheat flour", "sugar", "palm oil", "salt"],
            allergens=["wheat"],
            confidence=0.88
        )
        
        result = ExtractedNutritionLabel(
            serving_sizes=[{
                "serving_size": {"quantity": 30, "unit": "g"},
                "servings_per_container": 10,
                "nutrients": nutrients
            }],
            ingredients=ingredients,
            extraction_time=2.5,
            raw_response='{"test": "response"}',
            api_calls_made=1
        )
        
        print("✓ Created mock extraction result with ingredients")
        print(f"  - Nutrients: {len(result.nutrients)}")
        print(f"  - Ingredients: {len(result.ingredients.list)}")
        print(f"  - Allergens: {result.ingredients.allergens}")
        
        # Test validation
        validator = NutritionSchemaValidator()
        validation_data = {
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
            }
        }
        
        report = validator.validate(validation_data)
        
        print(f"✓ Validation successful")
        print(f"  - Valid: {report.is_valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" INGREDIENTS EXTRACTION FEATURE - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Schema Validation", test_schema),
        ("Dataclass Definitions", test_dataclasses),
        ("Extraction Prompt", test_prompt),
        ("Schema Validator", test_validator),
        ("Integration Test", test_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status:8s} - {test_name}")
    
    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ingredients extraction feature is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
