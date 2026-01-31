"""
Streamlit Web UI Application
=============================
Interactive web interface for nutrition label extraction and comparison.

Features:
- Image upload and quality assessment
- Real-time VLM-based extraction
- OCR baseline comparison
- Interactive results visualization
- Batch processing support

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any
import logging
import pandas as pd  

# Import modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.image_processor import ImageQualityAssessor, ImagePreprocessor, quick_quality_check
from src.vlm_extractor import NutritionVLMExtractor
from src.schema_validator import NutritionSchemaValidator
from src.evaluation import NutritionEvaluator
from src import utils
import config

# Setup logging
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title=config.STREAMLIT_CONFIG["page_title"],
    page_icon=config.STREAMLIT_CONFIG["page_icon"],
    layout=config.STREAMLIT_CONFIG["layout"],
    initial_sidebar_state=config.STREAMLIT_CONFIG["initial_sidebar_state"]
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# PAGE: HOME / DEMO
# ============================================================================

def page_home():
    """Main demo page for single image extraction."""
    st.title("🍎 Nutrition Label Extractor")
    st.subheader("Vision-Language Model Based Extraction System")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown("### Model Settings")
        min_confidence = st.slider(
            "Minimum Confidence Threshold",
            0.0, 1.0, config.MIN_CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Filter out nutrients with confidence below this threshold"
        )
        
        st.markdown("### Quality Assessment")
        skip_quality_check = st.checkbox(
            "Skip quality assessment",
            value=False,
            help="Process image even if quality is below threshold"
        )
        
        st.markdown("### Display Options")
        show_raw_response = st.checkbox(
            "Show raw VLM response",
            value=False,
            help="Display the complete response from Gemini Vision API"
        )
        
        show_detailed_metrics = st.checkbox(
            "Show detailed metrics",
            value=True,
            help="Display comprehensive evaluation metrics"
        )
    
    # Main content
    st.markdown("---")
    
    # Upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Upload Nutrition Label Image")
        st.info("📸 Supported formats: JPG, PNG (max 10MB)")
        
        uploaded_file = st.file_uploader(
            "Choose image file",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### Quick Info")
        st.write("""
        - **Input**: Photo of nutrition label
        - **Output**: Structured nutrition data (JSON)
        - **Method**: Gemini Vision API
        - **Processing**: ~2-5 seconds per image
        """)
    
    st.markdown("---")
    
    # Process image
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Display image
            image = cv2.imread(temp_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Original Image")
                st.image(image_rgb, use_column_width=True)
            
            with col2:
                st.subheader("Image Information")
                height, width = image.shape[:2]
                file_size = Path(temp_path).stat().st_size / 1024
                
                st.write(f"**Dimensions:** {width}×{height} px")
                st.write(f"**File Size:** {file_size:.1f} KB")
                st.write(f"**Format:** {uploaded_file.type}")
            
            st.markdown("---")
            
            # Quality assessment
            st.subheader("🔍 Quality Assessment")
            
            with st.spinner("Assessing image quality..."):
                assessor = ImageQualityAssessor()
                quality_score = assessor.assess(temp_path)
            
            # Display quality metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Score", f"{quality_score.overall_score:.2f}", 
                         "✓" if quality_score.is_valid else "✗")
            
            with col2:
                st.metric("Brightness", f"{quality_score.brightness:.2f}", 
                         delta=None)
            
            with col3:
                st.metric("Contrast", f"{quality_score.contrast:.2f}",
                         delta=None)
            
            with col4:
                st.metric("Sharpness", f"{quality_score.blur:.2f}",
                         delta=None)
            
            # Show quality issues as informational only
            if quality_score.issues:
                st.info("ℹ️ Quality Analysis:\n" + "\n".join(
                    f"- {issue}" for issue in quality_score.issues
                ) + "\n\n*Proceeding with extraction anyway. Gemini can handle low-quality images.*")
            
            # Always allow extraction (Gemini Vision is robust enough)
            st.markdown("---")
            st.subheader("📋 Nutrition Extraction")
            
            if st.button("Extract Nutrition Information", type="primary", use_container_width=True):
                    with st.spinner("🤖 Extracting with Gemini Vision..."):
                        try:
                            # Extract
                            extractor = NutritionVLMExtractor()
                            extracted = extractor.extract(temp_path)
                            
                            # Validate
                            validator = NutritionSchemaValidator()
                            validation_report = validator.validate({
                                "serving_size": extracted.serving_size,
                                "servings_per_container": extracted.servings_per_container,
                                "nutrients": [
                                    {
                                        "name": n.name,
                                        "value": n.value,
                                        "unit": n.unit,
                                        "per_rda": n.per_rda,
                                        "confidence": n.confidence
                                    }
                                    for n in extracted.nutrients
                                ],
                                "ingredients": {
                                    "list": extracted.ingredients.list,
                                    "allergens": extracted.ingredients.allergens,
                                    "confidence": extracted.ingredients.confidence
                                } if extracted.ingredients else None
                            })
                            
                            # Apply confidence filtering
                            filtered_nutrients = [
                                n for n in extracted.nutrients
                                if n.confidence >= min_confidence
                            ]
                            
                            st.success(f"✓ Successfully extracted {len(filtered_nutrients)} nutrients")

                            # INSERT THE NEW CODE HERE:
                            if hasattr(extracted, 'token_usage'):
                                st.info(f"🪙 **Token Usage:** Input: {extracted.token_usage.input_tokens} | Output: {extracted.token_usage.output_tokens} | Total: {extracted.token_usage.total_tokens}")

                            # Display results
                            st.markdown("---")
                            st.subheader("📊 Extraction Results")
                            
                            # ----------------------------------------------------------------------------
# REPLACEMENT CODE FOR SERVING SIZE DISPLAY
# ----------------------------------------------------------------------------
                            
                            # Display Serving Information
                            if extracted.serving_size:
                                st.markdown("### Serving Information")
                                col1, col2 = st.columns(2)
                                with col1:
                                    qty = extracted.serving_size.get('quantity', 'N/A')
                                    unit = extracted.serving_size.get('unit', '')
                                    st.write(f"**Serving Size:** {qty} {unit}")
                                with col2:
                                    st.write(f"**Servings per Container:** {extracted.servings_per_container or 'N/A'}")
                                st.markdown("---")

                            # Display Nutrients
                            st.subheader("📊 Nutrients")
                            
                            # Filter nutrients by confidence
                            filtered_nutrients = [
                                n for n in extracted.nutrients
                                if n.confidence >= min_confidence
                            ]
                            
                            st.write(f"Found {len(filtered_nutrients)} nutrients (min confidence: {min_confidence:.0%})")
                            
                            nutrient_data = []
                            for nutrient in filtered_nutrients:
                                nutrient_data.append({
                                    "Nutrient": nutrient.name,
                                    "Value": f"{nutrient.value}" if nutrient.value is not None else "N/A",
                                    "Unit": nutrient.unit or "",
                                    "%RDA": f"{nutrient.per_rda}%" if nutrient.per_rda is not None else "N/A",                                    "Confidence": f"{nutrient.confidence:.1%}"
                                })
                            
                            if nutrient_data:
                                st.dataframe(nutrient_data, use_container_width=True, hide_index=True)
                            else:
                                st.warning("No nutrients extracted with sufficient confidence.")
# ----------------------------------------------------------------------------

                            # Display ingredients if available
                            if extracted.ingredients and extracted.ingredients.list:
                                st.markdown("---")
                                st.subheader("🦾 Ingredients List")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Total Ingredients:** {len(extracted.ingredients.list)}")
                                with col2:
                                    st.write(f"**Confidence:** {extracted.ingredients.confidence:.1%}")
                                
                                # Display ingredients
                                st.markdown("**Ingredients (in order of predominance):**")
                                ingredients_text = ", ".join(extracted.ingredients.list)
                                st.info(ingredients_text)
                                
                                # Display allergens if available
                                if extracted.ingredients.allergens:
                                    st.markdown("**⚠️ Allergens Detected:**")
                                    allergen_cols = st.columns(min(len(extracted.ingredients.allergens), 4))
                                    for idx, allergen in enumerate(extracted.ingredients.allergens):
                                        with allergen_cols[idx % len(allergen_cols)]:
                                            st.warning(f"🚨 {allergen}")
                            
                            # Validation report
                            if not validation_report.is_valid:
                                st.markdown("---")
                                st.subheader("⚠️ Validation Warnings")
                                if validation_report.errors:
                                    for error in validation_report.errors:
                                        st.write(f"- **{error.field}**: {error.error}")
                                if validation_report.warnings:
                                    for warning in validation_report.warnings:
                                        st.write(f"- {warning}")
                            
                            # Detailed metrics
                            if show_detailed_metrics:
                                st.markdown("---")
                                st.subheader("📈 Detailed Metrics")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Extraction Time", f"{extracted.extraction_time:.2f}s")
                                with col2:
                                    st.metric("Total Nutrients", len(extracted.nutrients))
                                with col3:
                                    avg_confidence = np.mean([n.confidence for n in extracted.nutrients])
                                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                                with col4:
                                    st.metric("API Calls", extracted.api_calls_made)
                            
                            # Show raw response if requested
                            if show_raw_response:
                                st.markdown("---")
                                st.subheader("📝 Raw VLM Response")
                                st.json(json.loads(extracted.raw_response))
                            
                            # Download results
                            st.markdown("---")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Prepare JSON for download
                                result_json = {
                                    "serving_size": extracted.serving_size,
                                    "servings_per_container": extracted.servings_per_container,
                                    "nutrients": [
                                        {
                                            "name": n.name,
                                            "value": n.value,
                                            "unit": n.unit,
                                            "per_rda": n.per_rda,
                                            "confidence": n.confidence
                                        }
                                        for n in filtered_nutrients
                                    ],
                                    "ingredients": {
                                        "list": extracted.ingredients.list,
                                        "allergens": extracted.ingredients.allergens,
                                        "confidence": extracted.ingredients.confidence
                                    } if extracted.ingredients else None
                                }
                                
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=json.dumps(result_json, indent=2),
                                    file_name="nutrition_extraction.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                            
                            with col2:
                                # Prepare CSV for download
                                csv_data = "Nutrient,Value,Unit,RDA%,Confidence\n"
                                for n in filtered_nutrients:
                                    csv_data += f"{n.name},{n.value},{n.unit},{n.per_rda},{n.confidence}\n"
                                
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_data,
                                    file_name="nutrition_extraction.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        
                        except Exception as e:
                            st.error(f"❌ Error during extraction: {str(e)}")
                            logger.exception("Extraction error")
        
        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except:
                pass


# ============================================================================
# PAGE: BATCH PROCESSING
# ============================================================================

# ... (imports remain the same) ...

def page_batch():
    st.title("🚀 Batch Processing & Evaluation")
    
    # --- SELECT MODE ---
    mode = st.radio("Select Input Mode", ["📂 Manual Upload", "💻 Local Folders (One-Click)"], horizontal=True)

    # =========================================================================
    # MODE 1: MANUAL UPLOAD (Keep existing logic)
    # =========================================================================
    if mode == "📂 Manual Upload":
        st.subheader("Manual File Selection")
        col1, col2 = st.columns(2)
        with col1:
            img_files = st.file_uploader("Select Images", accept_multiple_files=True, key="imgs")
        with col2:
            gt_files = st.file_uploader("Select JSONs", accept_multiple_files=True, key="gts")
            
        if st.button("Start Manual Batch", type="primary"):
            # Reuse the processing logic function (defined below)
            process_batch_data(img_files, gt_files, is_local=False)

    # =========================================================================
    # MODE 2: LOCAL FOLDERS (New Feature)
    # =========================================================================
    else:
        st.subheader("Automated Folder Processing")
        st.info("This will automatically match images in 'complex_images' with JSONs in 'nutrients' and 'ingredients' folders.")
        
        c1, c2, c3 = st.columns(3)
        img_dir = c1.text_input("Images Folder", "complex_Images")
        nut_dir = c2.text_input("Nutrients GT Folder", "Nutrients")
        ing_dir = c3.text_input("Ingredients GT Folder", "Ingredients")
        
        if st.button("🚀 Process All Images", type="primary"):
            import os
            import glob
            
            # 1. Gather Files
            # Get all images (jpg, png, jpeg)
            image_paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                image_paths.extend(glob.glob(os.path.join(img_dir, ext)))
            
            if not image_paths:
                st.error(f"No images found in folder: {img_dir}")
                return
                
            st.success(f"Found {len(image_paths)} images. Looking for matching JSONs...")
            
            # 2. Gather GT Files (Auto-match)
            gt_files_list = []
            # We simply list all json files in the GT directories
            gt_files_list.extend(glob.glob(os.path.join(nut_dir, "*.json")))
            gt_files_list.extend(glob.glob(os.path.join(ing_dir, "*.json")))
            
            # 3. Process
            process_batch_data(image_paths, gt_files_list, is_local=True)

# =============================================================================
# SHARED PROCESSING LOGIC
# =============================================================================
# =============================================================================
# SHARED PROCESSING LOGIC (DEBUG VERSION)
# =============================================================================
def process_batch_data(image_inputs, gt_inputs, is_local=False):
    """
    Handles batch processing with DEBUG LOGGING to fix the file path error.
    """
    import json
    import time
    import pandas as pd
    import os
    import numpy as np
    import cv2
    
    # ----------------- DEBUG LOG -----------------
    print("\n" + "="*50)
    print("STARTING BATCH PROCESS (DEBUG MODE)")
    print("="*50 + "\n")
    # ---------------------------------------------
    
    extractor = NutritionVLMExtractor()
    preprocessor = ImagePreprocessor()
    evaluator = NutritionEvaluator()
    
    # 1. Map Ground Truth
    gt_map = {}
    
    def read_json(item):
        if is_local:
            with open(item, 'r', encoding='utf-8') as f: return json.load(f)
        return json.load(item)

    def get_name(item):
        return os.path.basename(item) if is_local else item.name

    if gt_inputs:
        for f in gt_inputs:
            fname = get_name(f)
            try:
                content = read_json(f)
                if "_nutrients" in fname:
                    clean_id = fname.split("_nutrients")[0]
                    if clean_id not in gt_map: gt_map[clean_id] = {}
                    gt_map[clean_id]['nutrients'] = content
                elif "_ingredients" in fname:
                    clean_id = fname.split("_ingredients")[0]
                    if clean_id not in gt_map: gt_map[clean_id] = {}
                    gt_map[clean_id]['ingredients'] = content
            except Exception as e:
                print(f"DEBUG: Error loading JSON {fname}: {e}")

    # 2. Processing Loop
    summary_results = []
    detailed_logs = []
    
    progress_bar = st.progress(0)
    status = st.empty()
    total_imgs = len(image_inputs)
    
    for i, img_obj in enumerate(image_inputs):
        img_name = get_name(img_obj)
        img_id = os.path.splitext(img_name)[0]
        status.text(f"Processing {i+1}/{total_imgs}: {img_name}...")
        
        # ----------------- DEBUG LOG -----------------
        print(f"--- Processing Image: {img_name} ---")
        # ---------------------------------------------
        
        # Save to temp
        if is_local:
            temp_path = img_obj
        else:
            temp_path = f"temp_{img_name}"
            with open(temp_path, "wb") as f: f.write(img_obj.getbuffer())

        proc_path = temp_path # Default to original if preprocessing fails

        try:
            t0 = time.time()
            
            # --- DEBUGGING PREPROCESSING RETURN TYPE ---
            raw_output = preprocessor.preprocess(temp_path)
            
            # Print type to console to debug
            print(f"DEBUG: Preprocessor returned type: {type(raw_output)}")
            
            # Logic to handle different return types
            if isinstance(raw_output, str):
                proc_path = raw_output
                print(f"DEBUG: Path is valid string: {proc_path}")
                
            elif isinstance(raw_output, (np.ndarray, list)):
                # IT IS AN IMAGE ARRAY -> SAVE IT MANUALLY
                print("DEBUG: Detected IMAGE DATA (Array/List). Saving manually...")
                
                # Convert list to numpy if needed
                if isinstance(raw_output, list):
                    raw_output = np.array(raw_output, dtype=np.uint8)
                    
                proc_path = f"proc_{os.path.basename(temp_path)}"
                cv2.imwrite(proc_path, raw_output)
                print(f"DEBUG: Saved array to {proc_path}")
                
            elif isinstance(raw_output, tuple):
                print("DEBUG: Detected TUPLE. Checking first element...")
                first = raw_output[0]
                if isinstance(first, str):
                    proc_path = first
                elif isinstance(first, (np.ndarray, list)):
                    print("DEBUG: Tuple contained array. Saving...")
                    if isinstance(first, list): first = np.array(first, dtype=np.uint8)
                    proc_path = f"proc_{os.path.basename(temp_path)}"
                    cv2.imwrite(proc_path, first)
            
            # Final Safety Check
            if not isinstance(proc_path, str) or len(str(proc_path)) > 255:
                # If proc_path is somehow still massive data, force reset to temp_path
                print("DEBUG: CRITICAL - proc_path seems invalid. Reverting to temp_path.")
                proc_path = temp_path

            t1 = time.time()
            
            # Extraction
            print(f"DEBUG: Extracting from file: {proc_path}")
            extracted = extractor.extract(proc_path)
            t2 = time.time()
            
            # Evaluation
            nut_acc = None
            rda_acc = None
            ing_acc = None
            gt_data = gt_map.get(img_id, {})
            
            if 'nutrients' in gt_data:
                nut_acc = evaluator.evaluate_nutrients(extracted.nutrients, gt_data['nutrients'])
                rda_acc = evaluator.evaluate_rda(extracted.nutrients, gt_data['nutrients'])
                
                # Detailed Nutrient Logging
                gt_nut_lookup = {n.get('nutrient', '').lower(): n.get('value') for n in gt_data['nutrients']}
                for n in extracted.nutrients:
                    n_name = n.name.lower()
                    gt_val = "MISSING"
                    status_match = "Missing in GT"
                    found_key = next((k for k in gt_nut_lookup if k in n_name or n_name in k), None)
                    if found_key:
                        gt_val = gt_nut_lookup[found_key]
                        try:
                            if float(n.value) == float(gt_val) or abs(float(n.value) - float(gt_val)) < 0.5:
                                status_match = "MATCH"
                            else:
                                status_match = "MISMATCH"
                        except:
                            status_match = "TYPE ERROR"
                    
                    detailed_logs.append({
                        "Image": img_id, "Type": "Nutrient Value", "Item": n.name,
                        "Predicted": n.value, "Ground Truth": gt_val, "Status": status_match,
                        "Confidence": n.confidence
                    })

            if 'ingredients' in gt_data and extracted.ingredients:
                ing_acc = evaluator.evaluate_ingredients(extracted.ingredients.list, gt_data['ingredients'])
                # Log Ingredients
                pred_set = set(x.lower() for x in extracted.ingredients.list)
                gt_set = set(x.lower() for x in gt_data['ingredients'])
                for item in pred_set.intersection(gt_set):
                    detailed_logs.append({"Image": img_id, "Type": "Ingredient", "Item": item, "Predicted": "Present", "Ground Truth": "Present", "Status": "MATCH"})

            # Summary Record
            summary_results.append({
                "Image ID": img_id,
                "Nutrients": len(extracted.nutrients),
                "Nutrient Acc": f"{nut_acc:.1%}" if nut_acc is not None else "N/A",
                "RDA Acc": f"{rda_acc:.1%}" if rda_acc is not None else "N/A",
                "Ingredient Acc": f"{ing_acc:.1%}" if ing_acc is not None else "N/A",
                "Tokens": extracted.token_usage.total_tokens,
                "Process Time": round(t1-t0, 2),
                "VLM Time": round(t2-t1, 2)
            })
            
        except Exception as e:
            st.error(f"Error processing {img_name}: {e}")
            print(f"DEBUG ERROR: {e}")
            import traceback
            traceback.print_exc() # Print full stack trace to console
        
        finally:
            if not is_local:
                try: Path(temp_path).unlink(missing_ok=True)
                except: pass
            try:
                if 'proc_path' in locals() and proc_path and os.path.exists(proc_path) and proc_path != temp_path:
                     if not (is_local and os.path.abspath(proc_path) == os.path.abspath(img_obj)):
                        os.remove(proc_path)
            except: pass

        progress_bar.progress((i + 1) / total_imgs)

    status.empty()
    st.success("✅ Batch Processing Complete!")
    
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_detailed = pd.DataFrame(detailed_logs)
        
        st.subheader("📈 Summary Metrics")
        st.dataframe(df_summary, use_container_width=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        summary_path = f"results/batch_summary_{timestamp}.csv"
        detailed_path = f"results/batch_detailed_{timestamp}.csv"
        df_summary.to_csv(summary_path, index=False)
        df_detailed.to_csv(detailed_path, index=False)
        
        st.info(f"Logs saved locally to: {summary_path}")
        
        col1, col2 = st.columns(2)
        col1.download_button("📥 Download Summary", df_summary.to_csv(index=False), f"summary_{timestamp}.csv")
        col2.download_button("📥 Download Detailed Logs", df_detailed.to_csv(index=False), f"detailed_{timestamp}.csv")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main app entry point."""
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🚀 Batch Processing"],
        label_visibility="collapsed"
    )
    
    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "**Nutrition Label Extraction System**\n\n"
        "A Vision-Language Model based pipeline for accurate extraction "
        "of nutrition information from food package labels.\n\n"
        "**Built with:** Gemini Vision API, Python, Streamlit"
    )
    
    # Route to pages
    if page == "🏠 Home":
        page_home()
    elif page == "🚀 Batch Processing":
        page_batch()


if __name__ == "__main__":
    main()
