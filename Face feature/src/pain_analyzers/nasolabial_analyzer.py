"""
Nasolabial Deepening Analyzer
Analyzes nasolabial fold depth changes indicating facial strain
"""

import numpy as np
from typing import Dict, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import calculate_distance, calculate_angle


class NasolabialAnalyzer:
    """Analyzes nasolabial fold deepening patterns"""
    
    def __init__(self, landmark_indices: Dict[str, list]):
        """
        Initialize the nasolabial analyzer
        
        Args:
            landmark_indices: Dictionary containing landmark indices for nasolabial area
        """
        self.nasolabial_left = landmark_indices.get('nasolabial_left', [48])
        self.nasolabial_right = landmark_indices.get('nasolabial_right', [278])
        
        # Additional landmarks for fold analysis
        self.nose_bottom_left = 98
        self.nose_bottom_right = 327
        self.mouth_corner_left = 61
        self.mouth_corner_right = 291
        self.cheek_left = 205
        self.cheek_right = 425
        
        # Baseline values
        self.baseline_left_depth = None
        self.baseline_right_depth = None
        self.baseline_fold_angle_left = None
        self.baseline_fold_angle_right = None
        
        # Historical data
        self.history = []
        self.history_size = 5  # Reduced from 10 for faster response
    
    def set_baseline(self, landmarks: np.ndarray):
        """
        Set baseline measurements from neutral expression
        
        Args:
            landmarks: Facial landmarks array (468, 3)
        """
        # Left side
        left_fold = landmarks[self.nasolabial_left[0]]
        nose_left = landmarks[self.nose_bottom_left]
        mouth_left = landmarks[self.mouth_corner_left]
        cheek_left = landmarks[self.cheek_left]
        
        # Calculate depth as distance from fold point to line between nose and mouth
        self.baseline_left_depth = self._calculate_fold_depth(left_fold, nose_left, mouth_left)
        self.baseline_fold_angle_left = calculate_angle(nose_left, left_fold, mouth_left)
        
        # Right side
        right_fold = landmarks[self.nasolabial_right[0]]
        nose_right = landmarks[self.nose_bottom_right]
        mouth_right = landmarks[self.mouth_corner_right]
        cheek_right = landmarks[self.cheek_right]
        
        self.baseline_right_depth = self._calculate_fold_depth(right_fold, nose_right, mouth_right)
        self.baseline_fold_angle_right = calculate_angle(nose_right, right_fold, mouth_right)
    
    def _calculate_fold_depth(self, fold_point: np.ndarray, nose_point: np.ndarray, 
                             mouth_point: np.ndarray) -> float:
        """
        Calculate the depth of nasolabial fold
        
        Args:
            fold_point: Point on the fold
            nose_point: Point near nose
            mouth_point: Point near mouth
        
        Returns:
            float: Depth measure
        """
        # Calculate perpendicular distance from fold point to line between nose and mouth
        # Using point-to-line distance formula
        line_vec = mouth_point - nose_point
        point_vec = fold_point - nose_point
        
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return 0
        
        line_unitvec = line_vec / line_len
        projection = np.dot(point_vec, line_unitvec)
        
        # Point on line closest to fold_point
        closest_point = nose_point + projection * line_unitvec
        
        # Distance is the depth
        depth = np.linalg.norm(fold_point - closest_point)
        
        return depth
    
    def analyze(self, landmarks: np.ndarray, use_baseline: bool = True) -> Dict[str, float]:
        """
        Analyze nasolabial deepening from landmarks
        
        Args:
            landmarks: Facial landmarks array (468, 3)
            use_baseline: Whether to use baseline comparison
        
        Returns:
            Dict containing analysis results
        """
        # Left side measurements
        left_fold = landmarks[self.nasolabial_left[0]]
        nose_left = landmarks[self.nose_bottom_left]
        mouth_left = landmarks[self.mouth_corner_left]
        cheek_left = landmarks[self.cheek_left]
        
        current_left_depth = self._calculate_fold_depth(left_fold, nose_left, mouth_left)
        current_fold_angle_left = calculate_angle(nose_left, left_fold, mouth_left)
        
        # Right side measurements
        right_fold = landmarks[self.nasolabial_right[0]]
        nose_right = landmarks[self.nose_bottom_right]
        mouth_right = landmarks[self.mouth_corner_right]
        cheek_right = landmarks[self.cheek_right]
        
        current_right_depth = self._calculate_fold_depth(right_fold, nose_right, mouth_right)
        current_fold_angle_right = calculate_angle(nose_right, right_fold, mouth_right)
        
        # Calculate strain score
        if use_baseline and self.baseline_left_depth is not None:
            # Deepening folds indicate strain
            left_depth_change = abs(current_left_depth - self.baseline_left_depth) / self.baseline_left_depth
            right_depth_change = abs(current_right_depth - self.baseline_right_depth) / self.baseline_right_depth
            
            left_angle_change = abs(current_fold_angle_left - self.baseline_fold_angle_left) / self.baseline_fold_angle_left
            right_angle_change = abs(current_fold_angle_right - self.baseline_fold_angle_right) / self.baseline_fold_angle_right
            
            strain_score = (
                (left_depth_change + right_depth_change) * 0.6 +
                (left_angle_change + right_angle_change) * 0.4
            ) / 2.0
        else:
            # Use absolute measurements
            # Deeper folds indicate more strain
            avg_depth = (current_left_depth + current_right_depth) / 2.0
            depth_factor = min(avg_depth * 30, 1.0)  # Scale depth
            
            # Angle changes
            avg_angle = (current_fold_angle_left + current_fold_angle_right) / 2.0
            angle_factor = 1.0 - (avg_angle / 180.0)
            
            strain_score = (depth_factor * 0.6 + angle_factor * 0.4)
        
        # Normalize and convert to percentage
        strain_score = np.clip(strain_score * 2.5, 0.0, 1.0)  # Increased sensitivity from 1.5 to 2.5
        strain_percentage = strain_score * 100
        
        # Apply smoothing with less weight to make it more responsive
        self.history.append(strain_score)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        # Use weighted average favoring recent frames
        if len(self.history) > 1:
            weights = np.linspace(0.5, 1.0, len(self.history))
            smoothed_score = np.average(self.history, weights=weights)
        else:
            smoothed_score = strain_score
        smoothed_percentage = smoothed_score * 100
        
        return {
            'strain_score': strain_score,
            'strain_percentage': strain_percentage,
            'smoothed_score': smoothed_score,
            'smoothed_percentage': smoothed_percentage,
            'left_depth': current_left_depth,
            'right_depth': current_right_depth,
            'left_angle': current_fold_angle_left,
            'right_angle': current_fold_angle_right,
            'has_baseline': self.baseline_left_depth is not None
        }
    
    def get_description(self, score: float) -> str:
        """
        Get text description of strain level
        
        Args:
            score: Strain score (0-1)
        
        Returns:
            str: Description of strain level
        """
        percentage = score * 100
        
        if percentage < 20:
            return "minimal facial strain"
        elif percentage < 40:
            return "slight nasolabial tension"
        elif percentage < 60:
            return "moderate facial strain"
        elif percentage < 80:
            return "significant strain (deepened folds)"
        else:
            return "severe facial strain (pronounced fold deepening)"
    
    def reset_history(self):
        """Clear historical data"""
        self.history = []
