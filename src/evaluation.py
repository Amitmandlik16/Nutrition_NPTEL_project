"""
Evaluation Module
=================
Compares extraction results against ground truth with flexible matching.
"""

import logging
import difflib
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class NutritionEvaluator:
    def __init__(self):
        # Synonyms for matching (e.g. Energy == Calories)
        self.synonyms = {
            "energy": ["calories", "cal", "energy", "kcal"],
            "protein": ["protein", "proteins"],
            "total fat": ["fat", "total fat"],
            "saturated fat": ["saturated fat", "saturated fatty acids"],
            "trans fat": ["trans fat", "trans fatty acids"],
            "carbohydrate": ["total carbohydrate", "carbs", "carbohydrate", "total carbohydrates"],
            "total sugars": ["sugar", "sugars", "total sugar", "total sugars"],
            "added sugars": ["added sugar", "added sugars"],
            "sodium": ["salt", "sodium"],
            "fiber": ["dietary fibre", "dietary fiber", "fiber"]
        }

    def _normalize(self, text: str) -> str:
        """Lowercase and remove special chars."""
        return re.sub(r'[^a-z0-9]', '', str(text).lower())

    def _names_match(self, pred_name: str, truth_name: str) -> bool:
        """Check if names are equivalent using synonyms or fuzzy match."""
        p = self._normalize(pred_name)
        t = self._normalize(truth_name)
        
        if p == t: return True
        
        for key, variants in self.synonyms.items():
            norm_variants = [self._normalize(v) for v in variants]
            if (p in norm_variants or p == self._normalize(key)) and \
               (t in norm_variants or t == self._normalize(key)):
                return True
                
        return difflib.SequenceMatcher(None, p, t).ratio() > 0.85

    def evaluate_nutrients(self, predicted: List[Any], ground_truth: List[Dict]) -> float:
        """Compare nutrient VALUES (mass/energy)."""
        if not ground_truth: return 0.0

        matches = 0
        total_comparable = 0
        
        # GT Map: {"energy": 28.8}
        gt_map = {}
        for item in ground_truth:
            name = item.get('nutrient')
            val = item.get('value')
            if name and val is not None and str(val).strip() != "":
                gt_map[name] = val

        for pred in predicted:
            matched_gt_name = None
            for gt_name in gt_map.keys():
                if self._names_match(pred.name, gt_name):
                    matched_gt_name = gt_name
                    break
            
            if matched_gt_name:
                total_comparable += 1
                try:
                    truth_val = float(gt_map[matched_gt_name])
                    curr_val = float(pred.value) if pred.value is not None else 0.0
                    
                    diff = abs(curr_val - truth_val)
                    # Logic: Exact 0 match, or 10% tolerance, or < 1.0 abs diff
                    if truth_val == 0:
                        if curr_val < 0.5: matches += 1
                    elif (diff / truth_val <= 0.10) or (diff <= 1.0):
                        matches += 1
                except (ValueError, TypeError):
                    pass

        return matches / total_comparable if total_comparable > 0 else 0.0

    def evaluate_rda(self, predicted: List[Any], ground_truth: List[Dict]) -> Optional[float]:
        """Compare RDA percentages."""
        if not ground_truth: return None

        matches = 0
        total_comparable = 0
        
        # GT Map for RDA: {"energy": "1.4%"}
        gt_map = {}
        for item in ground_truth:
            name = item.get('nutrient')
            rda = item.get('rda')
            # Only map if RDA is present and not empty string
            if name and rda and str(rda).strip() != "":
                gt_map[name] = rda

        for pred in predicted:
            matched_gt_name = None
            for gt_name in gt_map.keys():
                if self._names_match(pred.name, gt_name):
                    matched_gt_name = gt_name
                    break
            
            if matched_gt_name:
                total_comparable += 1
                try:
                    # Clean GT: "1.4%" -> 1.4
                    gt_str = str(gt_map[matched_gt_name]).replace('%', '').strip()
                    truth_val = float(gt_str)
                    
                    # Get Pred: Gemini extracts 1.4 directly
                    curr_val = float(pred.per_rda) if pred.per_rda is not None else 0.0
                    
                    diff = abs(curr_val - truth_val)
                    
                    # Logic: 10% tolerance or within 2 percentage points (e.g. 5% vs 7% is ok)
                    if truth_val == 0:
                        if curr_val < 1.0: matches += 1
                    elif (diff / truth_val <= 0.10) or (diff <= 2.0):
                        matches += 1
                except (ValueError, TypeError):
                    pass

        # If no items in GT had RDA, return None (N/A) instead of 0%
        return matches / total_comparable if total_comparable > 0 else None

    def evaluate_ingredients(self, pred_list: List[str], gt_list: List[str]) -> float:
        """Jaccard similarity for ingredients."""
        if not gt_list: return 1.0 if not pred_list else 0.0
        if not pred_list: return 0.0
        
        p_set = {self._normalize(x) for x in pred_list}
        t_set = {self._normalize(x) for x in gt_list}
        
        intersection = len(p_set.intersection(t_set))
        union = len(p_set.union(t_set))
        
        return intersection / union if union > 0 else 0.0