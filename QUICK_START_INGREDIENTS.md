# Quick Start Guide - Ingredients Extraction

## Overview
The Nutrition NPTEL project now extracts **both nutrition facts AND ingredients** from food package labels in a single pass!

## What's New? ✨

### Before
```json
{
  "serving_size": {...},
  "nutrients": [...]
}
```

### After
```json
{
  "serving_size": {...},
  "nutrients": [...],
  "ingredients": {
    "list": ["wheat flour", "sugar", "salt"],
    "allergens": ["wheat"],
    "confidence": 0.92
  }
}
```

## Usage Examples

### 1. Command Line

```bash
# Run extraction on an image
python main.py path/to/food_label.jpg

# Example output:
# ============================================================
# STEP 3: VLM-Based Extraction
# ============================================================
# ✓ Extraction complete in 2.43 seconds
#   Nutrients found: 12
#   Ingredients found: 8
#   Allergens identified: 2
#
#   Extracted nutrients:
#     1. Energy               250 kcal            [95.0% confidence]
#     2. Protein              8 g                 [92.0% confidence]
#     ...
#
#   Ingredients list (in order):
#     1. Whole grain wheat
#     2. Sugar
#     3. Palm oil
#     4. Salt
#     5. Soy lecithin
#     ...
#
#   ⚠️  Allergens detected:
#     • Wheat
#     • Soy
```

### 2. Web Interface (Streamlit)

```bash
streamlit run streamlit_app.py
```

Then:
1. Upload a food label image
2. Click "Extract Nutrition Information"
3. View results with:
   - 📊 Nutrition facts table
   - 🧂 Ingredients list
   - ⚠️ Allergen warnings

### 3. Python API

```python
from src.vlm_extractor import NutritionVLMExtractor

# Initialize extractor
extractor = NutritionVLMExtractor()

# Extract from image
result = extractor.extract("food_label.jpg")

# Access nutrition data
print(f"Nutrients: {len(result.nutrients)}")
for nutrient in result.nutrients:
    print(f"  {nutrient.name}: {nutrient.value} {nutrient.unit}")

# Access ingredients
if result.ingredients:
    print(f"\nIngredients: {len(result.ingredients.list)}")
    for ingredient in result.ingredients.list:
        print(f"  - {ingredient}")
    
    # Check allergens
    if result.ingredients.allergens:
        print(f"\nAllergens:")
        for allergen in result.ingredients.allergens:
            print(f"  ⚠️  {allergen}")
    
    print(f"\nConfidence: {result.ingredients.confidence:.1%}")
```

### 4. Save to JSON

```python
from src.vlm_extractor import NutritionVLMExtractor
import json

extractor = NutritionVLMExtractor()
result = extractor.extract("food_label.jpg")

# Prepare data
output_data = {
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

# Save
with open("output.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("✓ Saved to output.json")
```

## Example Output

```json
{
  "serving_size": {
    "quantity": 30,
    "unit": "g"
  },
  "servings_per_container": 10,
  "nutrients": [
    {
      "name": "Energy",
      "value": 150,
      "unit": "kcal",
      "per_rda": 7.5,
      "confidence": 0.95
    },
    {
      "name": "Protein",
      "value": 3,
      "unit": "g",
      "per_rda": 6,
      "confidence": 0.92
    }
  ],
  "ingredients": {
    "list": [
      "Whole grain wheat",
      "Sugar",
      "Palm oil",
      "Salt",
      "Soy lecithin (emulsifier)",
      "Natural flavor",
      "Vitamin E (mixed tocopherols)"
    ],
    "allergens": [
      "Wheat",
      "Soy"
    ],
    "confidence": 0.88
  }
}
```

## Tips for Best Results

### Image Quality
- ✅ **Good**: Clear, well-lit photos showing both nutrition facts AND ingredients section
- ✅ **Good**: High resolution (>800px width)
- ✅ **Good**: Straight-on angle, not skewed
- ❌ **Avoid**: Blurry images
- ❌ **Avoid**: Photos where ingredients are cut off or not visible

### What Gets Extracted

The system extracts:
1. **Complete ingredient list** - in order of predominance
2. **Sub-ingredients** - e.g., "wheat flour (wheat, malted barley)"
3. **Common allergens** - milk, eggs, peanuts, tree nuts, fish, shellfish, soy, wheat
4. **Confidence score** - how confident the AI is about the extraction

### Handling Missing Data

If ingredients aren't visible in the image:
```json
{
  "ingredients": {
    "list": [],
    "allergens": [],
    "confidence": 0.0
  }
}
```

## Common Use Cases

### 1. Allergen Checking
```python
def check_allergens(image_path, user_allergens):
    """Check if product contains user's allergens."""
    extractor = NutritionVLMExtractor()
    result = extractor.extract(image_path)
    
    if not result.ingredients:
        return "Ingredients not found in image"
    
    found_allergens = []
    for allergen in user_allergens:
        if any(allergen.lower() in a.lower() 
               for a in result.ingredients.allergens):
            found_allergens.append(allergen)
    
    if found_allergens:
        return f"⚠️  WARNING: Contains {', '.join(found_allergens)}"
    else:
        return "✓ Safe to consume"

# Usage
print(check_allergens("cereal.jpg", ["peanuts", "soy"]))
# Output: ⚠️  WARNING: Contains soy
```

### 2. Ingredient Count Analysis
```python
def analyze_ingredients(image_path):
    """Analyze ingredient complexity."""
    extractor = NutritionVLMExtractor()
    result = extractor.extract(image_path)
    
    if not result.ingredients:
        return "No ingredients found"
    
    count = len(result.ingredients.list)
    
    if count <= 5:
        return f"Simple product ({count} ingredients)"
    elif count <= 15:
        return f"Moderate complexity ({count} ingredients)"
    else:
        return f"Complex product ({count} ingredients)"

print(analyze_ingredients("snack.jpg"))
# Output: Moderate complexity (8 ingredients)
```

### 3. Export to Database
```python
import sqlite3
from src.vlm_extractor import NutritionVLMExtractor

def save_to_database(image_path, product_name):
    """Save extracted data to SQLite database."""
    extractor = NutritionVLMExtractor()
    result = extractor.extract(image_path)
    
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    
    # Save product
    cursor.execute("""
        INSERT INTO products (name, serving_size, confidence)
        VALUES (?, ?, ?)
    """, (product_name, 
          result.serving_size.get('quantity') if result.serving_size else None,
          result.ingredients.confidence if result.ingredients else 0))
    
    product_id = cursor.lastrowid
    
    # Save ingredients
    if result.ingredients:
        for ingredient in result.ingredients.list:
            cursor.execute("""
                INSERT INTO ingredients (product_id, name)
                VALUES (?, ?)
            """, (product_id, ingredient))
    
    conn.commit()
    conn.close()
    print(f"✓ Saved {product_name} to database")
```

## Troubleshooting

### Issue: Empty ingredients list
**Cause**: Ingredients section not visible in image  
**Solution**: Capture a photo that includes the full ingredients panel

### Issue: Low confidence score
**Cause**: Poor image quality or unclear text  
**Solution**: Retake with better lighting and higher resolution

### Issue: Missing allergens
**Cause**: AI didn't identify them or they're not in the common list  
**Solution**: Always manually verify allergen information for safety

### Issue: Incomplete ingredient list
**Cause**: Text cut off in image or too small to read  
**Solution**: Ensure entire ingredients section is visible and in focus

## Need Help?

- Read [INGREDIENTS_FEATURE.md](INGREDIENTS_FEATURE.md) for detailed documentation
- Check [README.md](README.md) for general setup instructions
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design details

## Verification

To verify the feature is working:
```bash
python verify_ingredients.py
```

Should output:
```
🎉 All checks passed! Ingredients feature is fully integrated.
```

---

**Happy extracting! 🍎**
