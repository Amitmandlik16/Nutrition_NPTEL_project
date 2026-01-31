# Ingredients Extraction Feature

## Overview
The Nutrition NPTEL project now includes a comprehensive **ingredients extraction feature** that works alongside the nutrition facts extraction. This feature uses the same Vision-Language Model (Gemini) to extract ingredient lists and identify allergens from food package labels.

## Features

### 1. **Ingredient List Extraction**
- Extracts ingredients in the exact order they appear on the package (order of predominance)
- Handles complex ingredient lists with sub-ingredients
- Example: "wheat flour (wheat, malted barley)" is preserved as-is

### 2. **Allergen Detection**
- Automatically identifies common allergens from the ingredients list
- Supported allergens:
  - Milk
  - Eggs
  - Peanuts
  - Tree nuts
  - Fish
  - Shellfish
  - Soy
  - Wheat

### 3. **Confidence Scoring**
- Each ingredients extraction includes a confidence score (0-1)
- Low confidence extractions can be filtered based on threshold

## Updated Files

### 1. **Schema** ([nutrition_schema.json](Nutrition%20NPTEL%20project/data/nutrition_schema.json))
```json
{
  "ingredients": {
    "type": "object",
    "description": "Ingredients information from the package",
    "properties": {
      "list": {
        "type": "array",
        "description": "List of ingredients in order of predominance",
        "items": {"type": "string"}
      },
      "allergens": {
        "type": "array",
        "description": "List of identified allergens",
        "items": {"type": "string"}
      },
      "confidence": {
        "type": "number",
        "description": "VLM confidence score (0-1)",
        "minimum": 0,
        "maximum": 1
      }
    }
  }
}
```

### 2. **Config** ([config.py](Nutrition%20NPTEL%20project/config.py))
Updated the extraction prompt to include:
- Instructions for ingredient extraction
- Allergen identification guidelines
- Order preservation requirements

### 3. **VLM Extractor** ([vlm_extractor.py](Nutrition%20NPTEL%20project/src/vlm_extractor.py))
Added:
- `ExtractedIngredients` dataclass
- Ingredients parsing in `_parse_response` method
- Updated `ExtractedNutritionLabel` to include ingredients field

### 4. **Main Script** ([main.py](Nutrition%20NPTEL%20project/main.py))
- Displays ingredients list after nutrients
- Shows allergen warnings
- Includes ingredients in saved JSON output

### 5. **Streamlit UI** ([streamlit_app.py](Nutrition%20NPTEL%20project/streamlit_app.py))
- New ingredients display section
- Allergen badges with visual warnings
- Ingredients included in downloadable JSON/CSV

### 6. **Validator** ([schema_validator.py](Nutrition%20NPTEL%20project/src/schema_validator.py))
Added:
- `_validate_ingredients` method
- Validation for ingredients list format
- Allergen data validation
- Confidence score checks

## Usage

### Command Line
```bash
python main.py path/to/food_label.jpg
```

**Output will include:**
```
Ingredients list (in order):
  1. Wheat flour
  2. Sugar
  3. Palm oil
  ...

⚠️  Allergens detected:
  • Wheat
  • Milk
```

### Streamlit Web UI
```bash
streamlit run streamlit_app.py
```

**Features in UI:**
- 🧂 Ingredients List section showing all ingredients
- ⚠️ Allergen warnings with visual badges
- Download results with ingredients included

### Python API
```python
from src.vlm_extractor import NutritionVLMExtractor

# Extract nutrition and ingredients
extractor = NutritionVLMExtractor()
result = extractor.extract("food_label.jpg")

# Access ingredients
if result.ingredients:
    print(f"Found {len(result.ingredients.list)} ingredients")
    print(f"Ingredients: {', '.join(result.ingredients.list)}")
    
    if result.ingredients.allergens:
        print(f"Allergens: {', '.join(result.ingredients.allergens)}")
    
    print(f"Confidence: {result.ingredients.confidence:.2%}")
```

## Data Structure

### Extracted Data Format
```json
{
  "serving_size": {
    "quantity": 30,
    "unit": "g"
  },
  "servings_per_container": 10,
  "nutrients": [...],
  "ingredients": {
    "list": [
      "Wheat flour (wheat, malted barley)",
      "Sugar",
      "Palm oil",
      "Salt",
      "Soy lecithin"
    ],
    "allergens": [
      "Wheat",
      "Soy"
    ],
    "confidence": 0.92
  }
}
```

## Validation Rules

The ingredients extraction is validated using the following rules:

1. **Format Validation**
   - `ingredients` must be a dictionary
   - `list` must be an array of strings
   - `allergens` must be an array (optional)
   - `confidence` must be between 0 and 1

2. **Content Validation**
   - Empty ingredients list triggers a warning
   - Empty/whitespace-only ingredients trigger warnings
   - Confidence below threshold shows warning

3. **Error Handling**
   - Invalid data types are reported as errors
   - Missing confidence defaults to 0.5
   - Missing allergens defaults to empty array

## Configuration

You can adjust the minimum confidence threshold in [config.py](Nutrition%20NPTEL%20project/config.py):

```python
# Minimum confidence to include extracted data
MIN_CONFIDENCE_THRESHOLD = 0.7
```

## Examples

### Example 1: Complete Extraction
**Input:** Image of cereal box with nutrition facts and ingredients

**Output:**
```
Nutrients found: 12
Ingredients found: 8
Allergens identified: 2

Ingredients list (in order):
  1. Whole grain oats
  2. Sugar
  3. Corn syrup
  4. Salt
  5. Calcium carbonate
  6. Vitamin E
  7. Niacinamide
  8. Vitamin B6

⚠️  Allergens detected:
  • Oats (may contain wheat)
```

### Example 2: No Ingredients Visible
If the ingredients section is not visible in the image:

**Output:**
```json
{
  "nutrients": [...],
  "ingredients": {
    "list": [],
    "allergens": [],
    "confidence": 0.0
  }
}
```

## Benefits

1. **Comprehensive Package Analysis**: Get both nutrition facts and ingredients in a single extraction
2. **Allergen Safety**: Automatic allergen identification helps with dietary restrictions
3. **Order Preservation**: Ingredients are maintained in order of predominance (regulatory requirement)
4. **High Accuracy**: Uses Gemini Vision for visual understanding of complex layouts
5. **Structured Output**: Clean JSON format for easy integration with other systems

## Best Practices

1. **Image Quality**: Ensure the ingredients section is clearly visible and in focus
2. **Lighting**: Good lighting helps the VLM read small text in ingredients lists
3. **Orientation**: Keep the image upright for best results
4. **Resolution**: Higher resolution images (>800px width) work better for small ingredient text

## Troubleshooting

### Empty Ingredients List
- **Cause**: Ingredients section not visible in image
- **Solution**: Capture a photo that includes the ingredients section

### Low Confidence Score
- **Cause**: Poor image quality, small text, or unclear layout
- **Solution**: Retake photo with better lighting and higher resolution

### Missing Allergens
- **Cause**: VLM didn't identify all allergens
- **Solution**: Always manually verify allergen information for safety-critical applications

## Future Enhancements

Potential improvements for the ingredients feature:
- [ ] Ingredient categorization (preservatives, additives, etc.)
- [ ] Nutritional scoring based on ingredients
- [ ] Multi-language ingredient support
- [ ] Ingredient substitution suggestions
- [ ] Detailed allergen analysis with cross-contamination warnings

## Testing

To test the new feature:

1. **Basic Test**:
   ```bash
   python main.py examples/sample_label.jpg
   ```

2. **Web UI Test**:
   ```bash
   streamlit run streamlit_app.py
   ```
   Upload an image and verify ingredients section appears

3. **Validation Test**:
   ```python
   from src.schema_validator import validate_nutrition_data
   
   data = {
       "nutrients": [...],
       "ingredients": {
           "list": ["ingredient1", "ingredient2"],
           "allergens": ["allergen1"],
           "confidence": 0.9
       }
   }
   
   report = validate_nutrition_data(data)
   print(report.is_valid)  # Should be True
   ```

## Support

For issues or questions about the ingredients extraction feature, please check:
- [README.md](Nutrition%20NPTEL%20project/README.md) for general setup
- [ARCHITECTURE.md](Nutrition%20NPTEL%20project/ARCHITECTURE.md) for system design
- Example outputs in the `results/` directory

---

**Note**: The ingredients extraction uses the same Gemini API as nutrition extraction, so there are no additional API costs or setup requirements.
