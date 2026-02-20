"""
Grimace Analyzer
Analyzes mouth and lip tension to detect grimacing patterns
"""

import numpy as np
from typing import Dict, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import calculate_distance, calculate_angle


class GrimaceAnalyzer:
    """Analyzes mouth grimacing and lip tension"""
    
    def __init__(self, landmark_indices: Dict[str, list]):
        """
        Initialize the grimace analyzer
        
        Args:
            landmark_indices: Dictionary containing landmark indices for mouth
        """
        self.mouth_corners = landmark_indices.get('mouth_corners', [61, 291])
        self.mouth_center = landmark_indices.get('mouth_center', [0, 17])
        
        # Additional mouth landmarks for better analysis
        self.upper_lip = [13, 14]  # Upper lip center
        self.lower_lip = [78, 308]  # Lower lip corners
        
        # Baseline values
        self.baseline_corner_distance = None
        self.baseline_vertical_distance = None
        self.baseline_lip_distance = None
        
        # Historical data
        self.history = []
        self.history_size = 5  # Reduced from 10 for faster response
    
    def set_baseline(self, landmarks: np.ndarray):
        """
        Set baseline measurements from neutral expression
        
        Args:
            landmarks: Facial landmarks array (468, 3)
        """
        left_corner = landmarks[self.mouth_corners[0]]
        right_corner = landmarks[self.mouth_corners[1]]
        top_center = landmarks[self.mouth_center[0]]
        bottom_center = landmarks[self.mouth_center[1]]
        
        # Horizontal distance between corners
        self.baseline_corner_distance = calculate_distance(left_corner, right_corner)
        
        # Vertical distance (mouth opening)
        self.baseline_vertical_distance = calculate_distance(top_center, bottom_center)
        
        # Lip tension
        upper = landmarks[self.upper_lip[0]]
        lower = landmarks[self.lower_lip[0]]
        self.baseline_lip_distance = calculate_distance(upper, lower)
    
    def analyze(self, landmarks: np.ndarray, use_baseline: bool = True) -> Dict[str, float]:
        """
        Analyze grimace intensity from landmarks
        
        Args:
            landmarks: Facial landmarks array (468, 3)
            use_baseline: Whether to use baseline comparison
        
        Returns:
            Dict containing analysis results
        """
        left_corner = landmarks[self.mouth_corners[0]]
        right_corner = landmarks[self.mouth_corners[1]]
        top_center = landmarks[self.mouth_center[0]]
        bottom_center = landmarks[self.mouth_center[1]]
        
        # Calculate current measurements
        current_corner_distance = calculate_distance(left_corner, right_corner)
        current_vertical_distance = calculate_distance(top_center, bottom_center)
        
        # Calculate mouth corner depression (y-coordinate comparison)
        corner_midpoint_y = (left_corner[1] + right_corner[1]) / 2
        center_y = (top_center[1] + bottom_center[1]) / 2
        corner_depression = max(0, corner_midpoint_y - center_y)
        
        # Lip tension
        upper = landmarks[self.upper_lip[0]]
        lower = landmarks[self.lower_lip[0]]
        current_lip_distance = calculate_distance(upper, lower)
        
        # Calculate grimace score
        if use_baseline and self.baseline_corner_distance is not None:
            # Grimacing typically involves:
            # 1. Corners pulled down or tight
            # 2. Reduced vertical opening
            # 3. Increased lip tension
            
            corner_change = abs(current_corner_distance - self.baseline_corner_distance) / self.baseline_corner_distance
            vertical_change = abs(current_vertical_distance - self.baseline_vertical_distance) / self.baseline_vertical_distance
            lip_change = abs(current_lip_distance - self.baseline_lip_distance) / self.baseline_lip_distance
            
            grimace_score = (corner_change * 0.4 + vertical_change * 0.3 + lip_change * 0.3)
        else:
            # Use absolute measurements
            # Tight mouth indicates grimacing
            corner_factor = 1.0 - min(current_corner_distance * 5, 1.0)
            vertical_factor = 1.0 - min(current_vertical_distance * 20, 1.0)
            depression_factor = min(corner_depression * 50, 1.0)
            
            grimace_score = (corner_factor * 0.35 + vertical_factor * 0.35 + depression_factor * 0.30)
        
        # Normalize and convert to percentage
        grimace_score = np.clip(grimace_score * 3.0, 0.0, 1.0)  # Increased sensitivity from 1.8 to 3.0
        grimace_percentage = grimace_score * 100
        
        # Apply smoothing with less weight to make it more responsive
        self.history.append(grimace_score)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        # Use weighted average favoring recent frames
        if len(self.history) > 1:
            weights = np.linspace(0.5, 1.0, len(self.history))
            smoothed_score = np.average(self.history, weights=weights)
        else:
            smoothed_score = grimace_score
        smoothed_percentage = smoothed_score * 100
        
        return {
            'grimace_score': grimace_score,
            'grimace_percentage': grimace_percentage,
            'smoothed_score': smoothed_score,
            'smoothed_percentage': smoothed_percentage,
            'corner_distance': current_corner_distance,
            'vertical_distance': current_vertical_distance,
            'corner_depression': corner_depression,
            'has_baseline': self.baseline_corner_distance is not None
        }
    
    def get_description(self, score: float) -> str:
        """
        Get text description of grimace level
        
        Args:
            score: Grimace score (0-1)
        
        Returns:
            str: Description of grimace level
        """
        percentage = score * 100
        
        if percentage < 20:
            return "relaxed expression"
        elif percentage < 40:
            return "slight mouth tension"
        elif percentage < 60:
            return "moderate grimacing"
        elif percentage < 80:
            return "significant grimacing (corners depressed)"
        else:
            return "severe grimacing (tight lips, corner tension)"
    
    def reset_history(self):
        """Clear historical data"""
        self.history = []
