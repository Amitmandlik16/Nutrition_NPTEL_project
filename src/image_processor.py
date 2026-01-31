"""
Image Processing Module
=======================
Handles image quality assessment and preprocessing for optimal VLM extraction.

Design Decision:
- Quality assessment comes FIRST to avoid wasting VLM API calls on bad images
- Preprocessing is LIGHT to preserve nutritional text readability
- Uses both statistical and perceptual quality metrics

Key Classes:
- ImageQualityAssessor: Evaluates image quality with detailed metrics
- ImagePreprocessor: Applies light preprocessing for clarity
"""

import cv2
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

import config

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Container for image quality assessment results."""
    overall_score: float  # 0-1
    brightness: float
    contrast: float
    blur: float
    resolution: float
    is_valid: bool
    issues: list  # List of quality issues found


class ImageQualityAssessor:
    """
    Assesses image quality across multiple dimensions.
    
    Why this matters:
    - Poor quality images lead to poor OCR/VLM results
    - Early detection saves API calls and computational resources
    - Provides detailed feedback for image collection
    """
    
    def __init__(self, config_dict: Dict = None):
        """
        Initialize quality assessor.
        
        Args:
            config_dict: Override default config (useful for testing)
        """
        self.config = config_dict or config.IMAGE_QUALITY
        
    def assess(self, image_path: str) -> QualityScore:
        """
        Comprehensive image quality assessment.
        
        Args:
            image_path: Path to image file
            
        Returns:
            QualityScore with detailed metrics
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            
            # Check individual quality metrics
            brightness_score = self._assess_brightness(img)
            contrast_score = self._assess_contrast(img)
            blur_score = self._assess_blur(img)
            resolution_score = self._assess_resolution(width, height)
            
            # Detect specific issues
            issues = self._detect_issues(
                brightness_score, contrast_score, blur_score, resolution_score
            )
            
            # Calculate weighted overall score
            overall_score = (
                brightness_score * 0.25 +
                contrast_score * 0.25 +
                blur_score * 0.25 +
                resolution_score * 0.25
            )
            
            is_valid = len(issues) == 0 and overall_score > 0.6
            
            return QualityScore(
                overall_score=overall_score,
                brightness=brightness_score,
                contrast=contrast_score,
                blur=blur_score,
                resolution=resolution_score,
                is_valid=is_valid,
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Error assessing image quality: {e}")
            raise
    
    def _assess_brightness(self, img: np.ndarray) -> float:
        """
        Assess if image is too dark or too bright (0-1).
        Nutrition labels need clear visibility of all text.
        """
        # Convert to grayscale and get average brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Ideal range: between min and max thresholds
        min_brightness = self.config["min_brightness"]
        max_brightness = self.config["max_brightness"]
        ideal_brightness = (min_brightness + max_brightness) / 2
        
        # Score based on proximity to ideal brightness
        if brightness < min_brightness or brightness > max_brightness:
            return 0.3  # Poor
        elif min_brightness + 20 < brightness < max_brightness - 20:
            return 1.0  # Excellent
        else:
            return 0.7  # Acceptable
    
    def _assess_contrast(self, img: np.ndarray) -> float:
        """
        Assess image contrast (0-1).
        High contrast = better text readability.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        
        # Normalize contrast score
        # Contrast typically ranges from 0-100, normalize to 0-1
        score = min(contrast / 100, 1.0)
        
        if contrast < self.config["min_contrast"]:
            return 0.3  # Poor contrast
        elif contrast > 50:
            return 1.0  # Excellent contrast
        else:
            return score
    
    def _assess_blur(self, img: np.ndarray) -> float:
        """
        Assess image blur using Laplacian variance method (0-1).
        Higher variance = less blur = sharper image.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = np.var(laplacian)
        
        threshold = self.config["blur_threshold"]
        
        if variance < threshold * 0.3:
            return 0.2  # Very blurry
        elif variance < threshold * 0.7:
            return 0.6  # Somewhat blurry
        elif variance < threshold:
            return 0.8  # Slightly blurry
        else:
            return 1.0  # Sharp
    
    def _assess_resolution(self, width: int, height: int) -> float:
        """
        Assess image resolution (0-1).
        VLM needs sufficient resolution to read text clearly.
        """
        min_width = self.config["min_width"]
        min_height = self.config["min_height"]
        
        if width < min_width or height < min_height:
            return 0.3  # Too small
        elif width > 3000 or height > 3000:
            return 0.7  # Very large (wastes computation)
        else:
            return 1.0  # Good resolution
    
    def _detect_issues(self, brightness: float, contrast: float, 
                      blur: float, resolution: float) -> list:
        """Identify specific quality issues for user feedback."""
        issues = []
        
        if brightness < 0.5:
            issues.append("Image too dark or too bright")
        if contrast < 0.5:
            issues.append("Low contrast - text may be hard to read")
        if blur < 0.6:
            issues.append("Image is blurry - consider retaking")
        if resolution < 0.5:
            issues.append("Image resolution too low")
        
        return issues


class ImagePreprocessor:
    """
    Applies LIGHT preprocessing to improve VLM extraction.
    
    Why "light"?
    - Heavy processing can remove nutritional text details
    - VLM is robust to moderate quality variations
    - Preprocessing goal: enhance readability without distortion
    """
    
    def __init__(self, config_dict: Dict = None):
        """Initialize preprocessor with config."""
        self.config = config_dict or config.PREPROCESSING
    
    def preprocess(self, image_path: str) -> Tuple[np.ndarray, str]:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (preprocessed_image, output_path)
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            logger.info(f"Preprocessing image: {image_path}")
            
            # Step 1: Resize to manageable dimensions
            img = self._resize_image(img)
            
            # Step 2: Apply CLAHE for contrast enhancement
            if self.config["apply_clahe"]:
                img = self._apply_clahe(img)
            
            # Step 3: Apply denoising
            if self.config["enable_denoising"]:
                img = self._denoise_image(img)
            
            # Step 4: Save preprocessed image
            output_path = self._save_preprocessed(image_path, img)
            
            logger.info(f"Preprocessing complete. Saved to: {output_path}")
            return img, output_path
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to fit within max dimensions while maintaining aspect ratio.
        Prevents wasting VLM computation on huge images.
        """
        height, width = img.shape[:2]
        max_width = self.config["resize_max_width"]
        max_height = self.config["resize_max_height"]
        
        # Check if resizing is needed
        if width <= max_width and height <= max_height:
            return img
        
        # Calculate scale factor
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        logger.debug(f"Resizing from ({width}x{height}) to ({new_width}x{new_height})")
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        
        Why CLAHE:
        - Enhances local contrast
        - Prevents over-amplification (unlike standard histogram equalization)
        - Helps with poor lighting conditions
        - Preserves edge details important for text
        """
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE only to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.config["clahe_clip_limit"],
            tileGridSize=(self.config["clahe_tile_size"], self.config["clahe_tile_size"])
        )
        l_channel = clahe.apply(l_channel)
        
        # Reconstruct image
        lab[:, :, 0] = l_channel
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        logger.debug("Applied CLAHE for contrast enhancement")
        return img
    
    def _denoise_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering for denoising.
        
        Why bilateral filter:
        - Preserves edges (text boundaries)
        - Removes noise without blurring text
        - Better than Gaussian blur for preserving details
        """
        denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        logger.debug("Applied bilateral filter for denoising")
        return denoised
    
    def _save_preprocessed(self, original_path: str, img: np.ndarray) -> str:
        """Save preprocessed image."""
        original_path = Path(original_path)
        output_dir = original_path.parent / "preprocessed"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"preprocessed_{original_path.name}"
        cv2.imwrite(str(output_path), img)
        
        return str(output_path)


def quick_quality_check(image_path: str, verbose: bool = False) -> bool:
    """
    Convenience function for quick quality assessment.
    
    Usage:
        if quick_quality_check("image.jpg"):
            # Process image
        else:
            # Image quality too poor
    """
    assessor = ImageQualityAssessor()
    score = assessor.assess(image_path)
    
    if verbose:
        print(f"Quality Score: {score.overall_score:.2f}")
        if score.issues:
            print("Issues:", ", ".join(score.issues))
    
    return score.is_valid
