"""
Brow Tension Analyzer
Analyzes eyebrow position and furrowing to detect pain indicators
"""

import numpy as np
from typing import Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils import calculate_distance, calculate_angle


class BrowAnalyzer:
    """Analyzes eyebrow tension and furrowing patterns"""
    
    def __init__(self, landmark_indices: Dict[str, list]):
        """
        Initialize the brow analyzer
        
        Args:
            landmark_indices: Dictionary containing landmark indices for eyebrows
        """
        self.eyebrow_left = landmark_indices.get('eyebrow_left', [70, 107])
        self.eyebrow_right = landmark_indices.get('eyebrow_right', [336, 300])
        
        # Baseline values (set during calibration)
        self.baseline_distance_left = None
        self.baseline_distance_right = None
        self.baseline_angle = None
        
        # Historical data for smoothing
        self.history = []
        self.history_size = 5  # Reduced from 10 for faster response
    
    def set_baseline(self, landmarks: np.ndarray):
        """
        Set baseline measurements from neutral expression
        
        Args:
            landmarks: Facial landmarks array (468, 3)
        """
        left_points = landmarks[self.eyebrow_left]
        right_points = landmarks[self.eyebrow_right]
        
        # Calculate baseline distances
        self.baseline_distance_left = calculate_distance(left_points[0], left_points[1])
        self.baseline_distance_right = calculate_distance(right_points[0], right_points[1])
        
        # Calculate baseline angle between eyebrows
        center_left = np.mean(left_points, axis=0)
        center_right = np.mean(right_points, axis=0)
        nose_bridge = landmarks[168]  # Nose bridge point
        
        self.baseline_angle = calculate_angle(center_left, nose_bridge, center_right)
    
    def analyze(self, landmarks: np.ndarray, use_baseline: bool = True) -> Dict[str, float]:
        """
        Analyze brow tension from landmarks
        
        Args:
            landmarks: Facial landmarks array (468, 3)
            use_baseline: Whether to use baseline comparison
        
        Returns:
            Dict containing analysis results
        """
        left_points = landmarks[self.eyebrow_left]
        right_points = landmarks[self.eyebrow_right]
        
        # Calculate current distances
        current_distance_left = calculate_distance(left_points[0], left_points[1])
        current_distance_right = calculate_distance(right_points[0], right_points[1])
        
        # Calculate current angle
        center_left = np.mean(left_points, axis=0)
        center_right = np.mean(right_points, axis=0)
        nose_bridge = landmarks[168]
        current_angle = calculate_angle(center_left, nose_bridge, center_right)
        
        # Calculate tension score
        if use_baseline and self.baseline_distance_left is not None:
            # Compare to baseline
            left_change = abs(current_distance_left - self.baseline_distance_left) / self.baseline_distance_left
            right_change = abs(current_distance_right - self.baseline_distance_right) / self.baseline_distance_right
            angle_change = abs(current_angle - self.baseline_angle) / self.baseline_angle
            
            # Furrowed brows typically move closer together and downward
            tension_score = (left_change + right_change + angle_change) / 3.0
        else:
            # Use absolute measurements
            # Lower angle typically indicates more tension
            angle_factor = 1.0 - (current_angle / 180.0)  # Normalize to 0-1
            
            # Closer eyebrows indicate tension
            avg_distance = (current_distance_left + current_distance_right) / 2.0
            distance_factor = 1.0 - min(avg_distance * 10, 1.0)  # Scale and cap at 1
            
            tension_score = (angle_factor * 0.6 + distance_factor * 0.4)
        
        # Normalize to 0-1 range and convert to percentage
        tension_score = np.clip(tension_score * 3.5, 0.0, 1.0)  # Increased sensitivity from 2.0 to 3.5
        tension_percentage = tension_score * 100
        
        # Apply smoothing with less weight to make it more responsive
        self.history.append(tension_score)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        # Use weighted average favoring recent frames
        if len(self.history) > 1:
            weights = np.linspace(0.5, 1.0, len(self.history))
            smoothed_score = np.average(self.history, weights=weights)
        else:
            smoothed_score = tension_score
        smoothed_percentage = smoothed_score * 100
        
        return {
            'tension_score': tension_score,
            'tension_percentage': tension_percentage,
            'smoothed_score': smoothed_score,
            'smoothed_percentage': smoothed_percentage,
            'left_distance': current_distance_left,
            'right_distance': current_distance_right,
            'angle': current_angle,
            'has_baseline': self.baseline_distance_left is not None
        }
    
    def get_description(self, score: float) -> str:
        """
        Get text description of tension level
        
        Args:
            score: Tension score (0-1)
        
        Returns:
            str: Description of tension level
        """
        percentage = score * 100
        
        if percentage < 20:
            return "minimal tension"
        elif percentage < 40:
            return "slight tension"
        elif percentage < 60:
            return "moderate tension"
        elif percentage < 80:
            return "significant tension"
        else:
            return "severe tension (furrowed brows)"
    
    def reset_history(self):
        """Clear historical data"""
        self.history = []
