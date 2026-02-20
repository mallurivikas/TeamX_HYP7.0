"""
Jaw Clench Analyzer
Analyzes jaw tension and clenching patterns
"""

import numpy as np
from typing import Dict, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import calculate_distance, calculate_angle


class JawAnalyzer:
    """Analyzes jaw clenching and tension"""
    
    def __init__(self, landmark_indices: Dict[str, list]):
        """
        Initialize the jaw analyzer
        
        Args:
            landmark_indices: Dictionary containing landmark indices for jaw
        """
        self.jaw_left = landmark_indices.get('jaw_left', [234, 93])
        self.jaw_right = landmark_indices.get('jaw_right', [454, 323])
        
        # Additional landmarks for comprehensive analysis
        self.jaw_bottom = 152  # Chin point
        self.jaw_left_angle = 172
        self.jaw_right_angle = 397
        
        # Baseline values
        self.baseline_left_distance = None
        self.baseline_right_distance = None
        self.baseline_jaw_width = None
        self.baseline_jaw_angle = None
        
        # Historical data
        self.history = []
        self.history_size = 5  # Reduced from 10 for faster response
    
    def set_baseline(self, landmarks: np.ndarray):
        """
        Set baseline measurements from neutral expression
        
        Args:
            landmarks: Facial landmarks array (468, 3)
        """
        left_points = landmarks[self.jaw_left]
        right_points = landmarks[self.jaw_right]
        jaw_bottom = landmarks[self.jaw_bottom]
        
        # Distance along jaw line
        self.baseline_left_distance = calculate_distance(left_points[0], left_points[1])
        self.baseline_right_distance = calculate_distance(right_points[0], right_points[1])
        
        # Jaw width
        left_angle = landmarks[self.jaw_left_angle]
        right_angle = landmarks[self.jaw_right_angle]
        self.baseline_jaw_width = calculate_distance(left_angle, right_angle)
        
        # Jaw angle (relaxed vs clenched)
        self.baseline_jaw_angle = calculate_angle(left_angle, jaw_bottom, right_angle)
    
    def analyze(self, landmarks: np.ndarray, use_baseline: bool = True) -> Dict[str, float]:
        """
        Analyze jaw clenching from landmarks
        
        Args:
            landmarks: Facial landmarks array (468, 3)
            use_baseline: Whether to use baseline comparison
        
        Returns:
            Dict containing analysis results
        """
        left_points = landmarks[self.jaw_left]
        right_points = landmarks[self.jaw_right]
        jaw_bottom = landmarks[self.jaw_bottom]
        
        # Current measurements
        current_left_distance = calculate_distance(left_points[0], left_points[1])
        current_right_distance = calculate_distance(right_points[0], right_points[1])
        
        # Jaw width
        left_angle = landmarks[self.jaw_left_angle]
        right_angle = landmarks[self.jaw_right_angle]
        current_jaw_width = calculate_distance(left_angle, right_angle)
        
        # Jaw angle
        current_jaw_angle = calculate_angle(left_angle, jaw_bottom, right_angle)
        
        # Calculate muscle tension (distance from jaw angle to cheek)
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        left_tension_distance = calculate_distance(left_angle, left_cheek)
        right_tension_distance = calculate_distance(right_angle, right_cheek)
        
        # Calculate clench score
        if use_baseline and self.baseline_left_distance is not None:
            # Clenching typically involves:
            # 1. Tighter jaw muscles (changed distances)
            # 2. Slightly wider jaw (muscle bulging)
            # 3. Changed jaw angle
            
            left_change = abs(current_left_distance - self.baseline_left_distance) / self.baseline_left_distance
            right_change = abs(current_right_distance - self.baseline_right_distance) / self.baseline_right_distance
            width_change = abs(current_jaw_width - self.baseline_jaw_width) / self.baseline_jaw_width
            angle_change = abs(current_jaw_angle - self.baseline_jaw_angle) / self.baseline_jaw_angle
            
            clench_score = (left_change * 0.25 + right_change * 0.25 + 
                          width_change * 0.25 + angle_change * 0.25)
        else:
            # Use absolute measurements
            # Tighter jaw indicates clenching
            avg_distance = (current_left_distance + current_right_distance) / 2.0
            distance_factor = 1.0 - min(avg_distance * 8, 1.0)
            
            # Wider jaw from muscle tension
            width_factor = min(current_jaw_width * 3, 1.0) - 0.5
            width_factor = max(0, width_factor)
            
            # Angle change (more acute when clenched)
            angle_factor = 1.0 - (current_jaw_angle / 180.0)
            
            clench_score = (distance_factor * 0.4 + width_factor * 0.3 + angle_factor * 0.3)
        
        # Normalize and convert to percentage
        clench_score = np.clip(clench_score * 3.5, 0.0, 1.0)  # Increased sensitivity from 2.0 to 3.5
        clench_percentage = clench_score * 100
        
        # Apply smoothing with less weight to make it more responsive
        self.history.append(clench_score)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        # Use weighted average favoring recent frames
        if len(self.history) > 1:
            weights = np.linspace(0.5, 1.0, len(self.history))
            smoothed_score = np.average(self.history, weights=weights)
        else:
            smoothed_score = clench_score
        smoothed_percentage = smoothed_score * 100
        
        return {
            'clench_score': clench_score,
            'clench_percentage': clench_percentage,
            'smoothed_score': smoothed_score,
            'smoothed_percentage': smoothed_percentage,
            'jaw_width': current_jaw_width,
            'jaw_angle': current_jaw_angle,
            'left_distance': current_left_distance,
            'right_distance': current_right_distance,
            'has_baseline': self.baseline_left_distance is not None
        }
    
    def get_description(self, score: float) -> str:
        """
        Get text description of clench level
        
        Args:
            score: Clench score (0-1)
        
        Returns:
            str: Description of clench level
        """
        percentage = score * 100
        
        if percentage < 20:
            return "jaw relaxed"
        elif percentage < 40:
            return "slight jaw tension"
        elif percentage < 60:
            return "moderate jaw clenching"
        elif percentage < 80:
            return "significant jaw tension (elevated muscle tension)"
        else:
            return "severe jaw clenching (tight jaw muscles)"
    
    def reset_history(self):
        """Clear historical data"""
        self.history = []
