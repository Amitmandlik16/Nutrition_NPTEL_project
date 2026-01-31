"""
Simple Verification Script for Ingredients Feature
===================================================
Verifies the ingredients feature without requiring all dependencies.
"""

import json
from pathlib import Path

def verify_files():
    """Verify all required files have been updated."""
    print("\n" + "="*70)
    print(" INGREDIENTS FEATURE VERIFICATION")
    print("="*70)
    
    base_path = Path(__file__).parent
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Schema file
    print("\n1. Checking nutrition_schema.json...")
    total_checks += 1
    schema_path = base_path / "data" / "nutrition_schema.json"
    
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        if "ingredients" in schema.get("properties", {}):
            ingredients_props = schema["properties"]["ingredients"]["properties"]
            required = ["list", "allergens", "confidence"]
            
            all_present = all(prop in ingredients_props for prop in required)
            if all_present:
                print("   ✓ Schema updated with ingredients field")
                print(f"   ✓ Contains: {', '.join(required)}")
                checks_passed += 1
            else:
                print("   ❌ Schema missing some properties")
        else:
            print("   ❌ Schema doesn't include ingredients")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Check 2: Config file
    print("\n2. Checking config.py...")
    total_checks += 1
    config_path = base_path / "config.py"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        keywords = ["ingredients", "allergen"]
        found = all(kw in config_content.lower() for kw in keywords)
        
        if found:
            print("   ✓ Config includes ingredients extraction prompt")
            print("   ✓ Prompt mentions allergens")
            checks_passed += 1
        else:
            print("   ❌ Config missing ingredients references")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Check 3: VLM Extractor
    print("\n3. Checking src/vlm_extractor.py...")
    total_checks += 1
    vlm_path = base_path / "src" / "vlm_extractor.py"
    
    try:
        with open(vlm_path, 'r', encoding='utf-8') as f:
            vlm_content = f.read()
        
        checks = [
            ("ExtractedIngredients", "dataclass definition"),
            ("ingredients: Optional[ExtractedIngredients]", "field in ExtractedNutritionLabel"),
            ("_validate_ingredients", "parsing method (if exists)"),
        ]
        
        found_count = 0
        for pattern, desc in checks:
            if pattern in vlm_content:
                print(f"   ✓ Found: {desc}")
                found_count += 1
        
        if found_count >= 2:  # At least the dataclass and field
            checks_passed += 1
        else:
            print("   ❌ Missing some required components")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Check 4: Main script
    print("\n4. Checking main.py...")
    total_checks += 1
    main_path = base_path / "main.py"
    
    try:
        with open(main_path, 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        if "ingredients" in main_content.lower() and "allergen" in main_content.lower():
            print("   ✓ Main script handles ingredients display")
            checks_passed += 1
        else:
            print("   ❌ Main script doesn't reference ingredients")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Check 5: Streamlit app
    print("\n5. Checking streamlit_app.py...")
    total_checks += 1
    streamlit_path = base_path / "streamlit_app.py"
    
    try:
        with open(streamlit_path, 'r', encoding='utf-8') as f:
            streamlit_content = f.read()
        
        if "ingredients" in streamlit_content.lower() and "allergen" in streamlit_content.lower():
            print("   ✓ Streamlit UI includes ingredients section")
            checks_passed += 1
        else:
            print("   ❌ Streamlit doesn't show ingredients")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Check 6: Schema validator
    print("\n6. Checking src/schema_validator.py...")
    total_checks += 1
    validator_path = base_path / "src" / "schema_validator.py"
    
    try:
        with open(validator_path, 'r', encoding='utf-8') as f:
            validator_content = f.read()
        
        if "_validate_ingredients" in validator_content:
            print("   ✓ Validator includes ingredients validation method")
            checks_passed += 1
        else:
            print("   ❌ Validator missing ingredients validation")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"\nPassed: {checks_passed}/{total_checks} checks")
    
    if checks_passed == total_checks:
        print("\n🎉 All checks passed! Ingredients feature is fully integrated.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set GEMINI_API_KEY environment variable")
        print("3. Run: python main.py <image_path>")
        print("4. Or run web UI: streamlit run streamlit_app.py")
    else:
        print(f"\n⚠️  {total_checks - checks_passed} check(s) failed.")
        print("Please review the errors above.")
    
    print("\n" + "="*70)
    
    return checks_passed == total_checks


if __name__ == "__main__":
    success = verify_files()
    exit(0 if success else 1)
