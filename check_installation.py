#!/usr/bin/env python3
"""
QUICK START GUIDE - Nutrition Label Extraction System
======================================================

This script validates your installation and runs a quick test.
"""

import sys
import os
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def check_installation():
    """Check if all required components are installed."""
    print_header("INSTALLATION CHECK")
    
    # Check Python version
    print(f"✓ Python version: {sys.version}")
    
    # Check required modules
    required_packages = [
        'google.generativeai',
        'PIL',
        'cv2',
        'pydantic',
        'jsonschema',
        'streamlit'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package:30s} installed")
        except ImportError:
            print(f"✗ {package:30s} MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"\nInstall with:")
        print(f"  pip install -r requirements.txt\n")
        return False
    else:
        print(f"\n✓ All packages installed!\n")
        return True

def check_api_key():
    """Check if API key is configured."""
    print_header("API KEY CHECK")
    
    # Check environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key and api_key != "your_api_key_here":
        print(f"✓ GEMINI_API_KEY set (length: {len(api_key)} chars)")
        return True
    else:
        print(f"✗ GEMINI_API_KEY not set")
        print(f"\nSet your API key:")
        print(f"  export GEMINI_API_KEY='your_key_here'")
        print(f"\n  Or edit config.py and set:")
        print(f"  GEMINI_API_KEY = 'your_key_here'\n")
        return False

def check_config():
    """Check configuration."""
    print_header("CONFIGURATION CHECK")
    
    try:
        import config
        
        print(f"✓ config.py loaded successfully")
        print(f"\nKey settings:")
        print(f"  Model:              {config.VLM_MODEL}")
        print(f"  Min Confidence:     {config.MIN_CONFIDENCE_THRESHOLD}")
        print(f"  Debug Mode:         {config.DEBUG_MODE}")
        print(f"  Save Results:       {config.SAVE_RESULTS}")
        print(f"  Results Directory:  {config.RESULTS_DIR}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading config.py: {e}\n")
        return False

def check_modules():
    """Check if all source modules exist."""
    print_header("SOURCE MODULES CHECK")
    
    modules = [
        'src/__init__.py',
        'src/image_processor.py',
        'src/vlm_extractor.py',
        'src/schema_validator.py',
        'src/evaluation.py',
        'src/utils.py',
    ]
    
    all_exist = True
    for module in modules:
        path = Path(module)
        if path.exists():
            size = path.stat().st_size
            print(f"✓ {module:35s} ({size:,} bytes)")
        else:
            print(f"✗ {module:35s} NOT FOUND")
            all_exist = False
    
    if all_exist:
        print(f"\n✓ All modules found!\n")
    else:
        print(f"\n✗ Some modules missing!\n")
    
    return all_exist

def print_next_steps():
    """Print next steps."""
    print_header("NEXT STEPS")
    
    print("1. GET API KEY")
    print("   Visit: https://cloud.google.com/docs/authentication/getting-started")
    print("   Create a Google Cloud project and enable Gemini Vision API")
    print()
    
    print("2. SET API KEY")
    print("   Option A: Environment variable")
    print("     export GEMINI_API_KEY='your_api_key'")
    print()
    print("   Option B: Edit config.py")
    print("     GEMINI_API_KEY = 'your_api_key'")
    print()
    
    print("3. RUN WEB UI (RECOMMENDED)")
    print("   streamlit run streamlit_app.py")
    print("   Then open: http://localhost:8501")
    print()
    
    print("4. OR RUN COMMAND LINE")
    print("   python main.py nutrition_label.jpg")
    print()
    
    print("5. READ DOCUMENTATION")
    print("   📖 README.md - Complete guide")
    print("   📖 PROJECT_OVERVIEW.txt - Quick overview")
    print("   📖 ARCHITECTURE.md - Technical design")
    print()

def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("  NUTRITION LABEL EXTRACTION SYSTEM")
    print("  Installation & Configuration Check")
    print("="*60)
    
    # Run checks
    checks = {
        "Python packages": check_installation(),
        "API Key": check_api_key(),
        "Configuration": check_config(),
        "Source modules": check_modules(),
    }
    
    # Summary
    print_header("SUMMARY")
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:30s}: {status}")
    
    print()
    
    if all_passed:
        print("✓ All checks passed! System is ready to use.\n")
        print_next_steps()
    else:
        print("⚠️  Some checks failed. Please fix the issues above.\n")
        print_next_steps()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
