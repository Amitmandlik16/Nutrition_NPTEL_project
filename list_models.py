"""
Script to list all available Gemini models for your API key.
Run this to see which models you can use.
"""
import google.generativeai as genai
from config import GEMINI_API_KEY

# Configure API
genai.configure(api_key=GEMINI_API_KEY)

print("=" * 80)
print("Available Gemini Models")
print("=" * 80)

try:
    # List all available models
    models = genai.list_models()
    
    vision_models = []
    text_models = []
    
    for model in models:
        print(f"\nModel: {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Description: {model.description}")
        print(f"  Supported Methods: {model.supported_generation_methods}")
        
        # Check if it supports generateContent (what we need)
        if 'generateContent' in model.supported_generation_methods:
            # Extract model name (remove 'models/' prefix)
            model_name = model.name.replace('models/', '')
            
            # Check if it's a vision model
            if 'vision' in model_name.lower() or 'pro' in model_name.lower():
                vision_models.append(model_name)
            else:
                text_models.append(model_name)
    
    print("\n" + "=" * 80)
    print("RECOMMENDED FOR VISION TASKS:")
    print("=" * 80)
    for model in vision_models:
        print(f"  ✓ {model}")
    
    if not vision_models:
        print("\nNo specific vision models found. Try these general models:")
        for model in text_models:
            print(f"  • {model}")
    
    print("\n" + "=" * 80)
    print("To use a model, update config.py:")
    print("  VLM_MODEL = 'model-name-from-above'")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ Error listing models: {e}")
    print("\nPossible issues:")
    print("  1. Invalid API key")
    print("  2. API not enabled in Google Cloud Console")
    print("  3. Network/firewall issues")
