"""
Eye Squint Analyzer
Analyzes eye aperture and squinting patterns
"""

import numpy as np
from typing import Dict, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import calculate_distance


class EyeAnalyzer:
    """Analyzes eye squinting patterns"""
    
    def __init__(self, landmark_indices: Dict[str, list]):
        """
        Initialize the eye analyzer
        
        Args:
            landmark_indices: Dictionary containing landmark indices for eyes
        """
        self.eye_left = landmark_indices.get('eye_left', [159, 145])
        self.eye_right = landmark_indices.get('eye_right', [386, 374])
        
        # Additional eye landmarks for better analysis
        self.left_upper = 159
        self.left_lower = 145
        self.right_upper = 386
        self.right_lower = 374
        
        # Additional horizontal landmarks
        self.left_inner = 133
        self.left_outer = 33
        self.right_inner = 362
        self.right_outer = 263
        
        # Baseline values
        self.baseline_left_aperture = None
        self.baseline_right_aperture = None
        self.baseline_left_width = None
        self.baseline_right_width = None
        
        # Historical data
        self.history = []
        self.history_size = 5  # Reduced from 10 for faster response
    
    def set_baseline(self, landmarks: np.ndarray):
        """
        Set baseline measurements from neutral expression
        
        Args:
            landmarks: Facial landmarks array (468, 3)
        """
        # Vertical aperture (eye height)
        left_upper = landmarks[self.left_upper]
        left_lower = landmarks[self.left_lower]
        right_upper = landmarks[self.right_upper]
        right_lower = landmarks[self.right_lower]
        
        self.baseline_left_aperture = calculate_distance(left_upper, left_lower)
        self.baseline_right_aperture = calculate_distance(right_upper, right_lower)
        
        # Horizontal width
        left_inner = landmarks[self.left_inner]
        left_outer = landmarks[self.left_outer]
        right_inner = landmarks[self.right_inner]
        right_outer = landmarks[self.right_outer]
        
        self.baseline_left_width = calculate_distance(left_inner, left_outer)
        self.baseline_right_width = calculate_distance(right_inner, right_outer)
    
    def analyze(self, landmarks: np.ndarray, use_baseline: bool = True) -> Dict[str, float]:
        """
        Analyze eye squinting from landmarks
        
        Args:
            landmarks: Facial landmarks array (468, 3)
            use_baseline: Whether to use baseline comparison
        
        Returns:
            Dict containing analysis results
        """
        # Get current measurements
        left_upper = landmarks[self.left_upper]
        left_lower = landmarks[self.left_lower]
        right_upper = landmarks[self.right_upper]
        right_lower = landmarks[self.right_lower]
        
        current_left_aperture = calculate_distance(left_upper, left_lower)
        current_right_aperture = calculate_distance(right_upper, right_lower)
        
        # Horizontal width
        left_inner = landmarks[self.left_inner]
        left_outer = landmarks[self.left_outer]
        right_inner = landmarks[self.right_inner]
        right_outer = landmarks[self.right_outer]
        
        current_left_width = calculate_distance(left_inner, left_outer)
        current_right_width = calculate_distance(right_inner, right_outer)
        
        # Calculate aspect ratio (height/width) for each eye
        left_aspect_ratio = current_left_aperture / current_left_width if current_left_width > 0 else 0
        right_aspect_ratio = current_right_aperture / current_right_width if current_right_width > 0 else 0
        
        # Calculate squint score
        if use_baseline and self.baseline_left_aperture is not None:
            # Squinting causes reduced aperture
            left_reduction = 1.0 - (current_left_aperture / self.baseline_left_aperture)
            right_reduction = 1.0 - (current_right_aperture / self.baseline_right_aperture)
            
            # Width might also decrease slightly
            left_width_change = abs(current_left_width - self.baseline_left_width) / self.baseline_left_width
            right_width_change = abs(current_right_width - self.baseline_right_width) / self.baseline_right_width
            
            squint_score = (
                (left_reduction + right_reduction) * 0.7 +
                (left_width_change + right_width_change) * 0.3
            ) / 2.0
        else:
            # Use absolute measurements
            # Lower aspect ratio indicates squinting
            avg_aspect_ratio = (left_aspect_ratio + right_aspect_ratio) / 2.0
            
            # Normal eye aspect ratio is around 0.25-0.35
            # Squinting reduces it to 0.15-0.20
            normal_ratio = 0.30
            squint_factor = max(0, normal_ratio - avg_aspect_ratio) / normal_ratio
            
            squint_score = squint_factor
        
        # Normalize and convert to percentage
        squint_score = np.clip(squint_score * 3.5, 0.0, 1.0)  # Increased sensitivity from 2.0 to 3.5
        squint_percentage = squint_score * 100
        
        # Apply smoothing with less weight to make it more responsive
        self.history.append(squint_score)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        # Use weighted average favoring recent frames
        if len(self.history) > 1:
            weights = np.linspace(0.5, 1.0, len(self.history))
            smoothed_score = np.average(self.history, weights=weights)
        else:
            smoothed_score = squint_score
        smoothed_percentage = smoothed_score * 100
        
        return {
            'squint_score': squint_score,
            'squint_percentage': squint_percentage,
            'smoothed_score': smoothed_score,
            'smoothed_percentage': smoothed_percentage,
            'left_aperture': current_left_aperture,
            'right_aperture': current_right_aperture,
            'left_aspect_ratio': left_aspect_ratio,
            'right_aspect_ratio': right_aspect_ratio,
            'has_baseline': self.baseline_left_aperture is not None
        }
    
    def get_description(self, score: float) -> str:
        """
        Get text description of squint level
        
        Args:
            score: Squint score (0-1)
        
        Returns:
            str: Description of squint level
        """
        percentage = score * 100
        
        if percentage < 20:
            return "eyes relaxed, normal aperture"
        elif percentage < 40:
            return "slight eye narrowing"
        elif percentage < 60:
            return "moderate squinting"
        elif percentage < 80:
            return "significant squinting (narrowed eyes)"
        else:
            return "severe squinting (eyes tightly narrowed)"
    
    def reset_history(self):
        """Clear historical data"""
        self.history = []
