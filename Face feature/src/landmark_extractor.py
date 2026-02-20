"""
Landmark Extractor
Utilities for extracting and processing specific facial landmarks
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import LANDMARK_INDICES
from src.utils import calculate_distance, calculate_angle, calculate_center


class LandmarkExtractor:
    """
    Extract and process specific facial landmarks for pain analysis
    """
    
    def __init__(self):
        """Initialize landmark extractor with predefined indices"""
        self.landmark_indices = LANDMARK_INDICES
    
    def get_eyebrow_landmarks(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract eyebrow landmarks
        
        Args:
            landmarks: Full landmark array (468, 3)
        
        Returns:
            Dict with 'left' and 'right' eyebrow landmarks
        """
        return {
            'left': landmarks[self.landmark_indices['eyebrow_left']],
            'right': landmarks[self.landmark_indices['eyebrow_right']]
        }
    
    def get_mouth_landmarks(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract mouth landmarks
        
        Args:
            landmarks: Full landmark array (468, 3)
        
        Returns:
            Dict with 'corners' and 'center' mouth landmarks
        """
        return {
            'corners': landmarks[self.landmark_indices['mouth_corners']],
            'center': landmarks[self.landmark_indices['mouth_center']]
        }
    
    def get_eye_landmarks(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract eye landmarks
        
        Args:
            landmarks: Full landmark array (468, 3)
        
        Returns:
            Dict with 'left' and 'right' eye landmarks
        """
        return {
            'left': landmarks[self.landmark_indices['eye_left']],
            'right': landmarks[self.landmark_indices['eye_right']]
        }
    
    def get_jaw_landmarks(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract jaw landmarks
        
        Args:
            landmarks: Full landmark array (468, 3)
        
        Returns:
            Dict with 'left' and 'right' jaw landmarks
        """
        return {
            'left': landmarks[self.landmark_indices['jaw_left']],
            'right': landmarks[self.landmark_indices['jaw_right']]
        }
    
    def get_nasolabial_landmarks(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract nasolabial landmarks
        
        Args:
            landmarks: Full landmark array (468, 3)
        
        Returns:
            Dict with 'left' and 'right' nasolabial landmarks
        """
        return {
            'left': landmarks[self.landmark_indices['nasolabial_left']],
            'right': landmarks[self.landmark_indices['nasolabial_right']]
        }
    
    def get_all_pain_landmarks(self, landmarks: np.ndarray) -> Dict[str, Dict]:
        """
        Extract all pain-related landmarks
        
        Args:
            landmarks: Full landmark array (468, 3)
        
        Returns:
            Dict containing all pain indicator landmarks
        """
        return {
            'eyebrows': self.get_eyebrow_landmarks(landmarks),
            'mouth': self.get_mouth_landmarks(landmarks),
            'eyes': self.get_eye_landmarks(landmarks),
            'jaw': self.get_jaw_landmarks(landmarks),
            'nasolabial': self.get_nasolabial_landmarks(landmarks)
        }
    
    def calculate_eyebrow_distance(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate distances between eyebrow landmarks
        
        Args:
            landmarks: Full landmark array (468, 3)
        
        Returns:
            Dict with distance measurements
        """
        eyebrows = self.get_eyebrow_landmarks(landmarks)
        
        left_distance = calculate_distance(eyebrows['left'][0], eyebrows['left'][1])
        right_distance = calculate_distance(eyebrows['right'][0], eyebrows['right'][1])
        
        # Distance between eyebrows (inner points)
        inter_eyebrow = calculate_distance(eyebrows['left'][1], eyebrows['right'][1])
        
        return {
            'left_span': left_distance,
            'right_span': right_distance,
            'inter_eyebrow': inter_eyebrow,
            'average_span': (left_distance + right_distance) / 2
        }
    
    def calculate_mouth_metrics(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate mouth-related metrics
        
        Args:
            landmarks: Full landmark array (468, 3)
        
        Returns:
            Dict with mouth measurements
        """
        mouth = self.get_mouth_landmarks(landmarks)
        
        # Distance between mouth corners
        corner_distance = calculate_distance(mouth['corners'][0], mouth['corners'][1])
        
        # Vertical mouth opening
        vertical_opening = calculate_distance(mouth['center'][0], mouth['center'][1])
        
        # Calculate mouth angle (corners relative to center)
        center_point = calculate_center(mouth['center'])
        
        # Check if corners are drooping (below center)
        left_droop = mouth['corners'][0][1] - center_point[1]
        right_droop = mouth['corners'][1][1] - center_point[1]
        
        return {
            'width': corner_distance,
            'height': vertical_opening,
            'left_corner_droop': left_droop,
            'right_corner_droop': right_droop,
            'average_droop': (left_droop + right_droop) / 2
        }
    
    def calculate_eye_aperture(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate eye opening measurements
        
        Args:
            landmarks: Full landmark array (468, 3)
        
        Returns:
            Dict with eye aperture measurements
        """
        eyes = self.get_eye_landmarks(landmarks)
        
        # Vertical eye opening (distance between upper and lower eyelid)
        left_aperture = calculate_distance(eyes['left'][0], eyes['left'][1])
        right_aperture = calculate_distance(eyes['right'][0], eyes['right'][1])
        
        return {
            'left_aperture': left_aperture,
            'right_aperture': right_aperture,
            'average_aperture': (left_aperture + right_aperture) / 2,
            'asymmetry': abs(left_aperture - right_aperture)
        }
    
    def calculate_jaw_tension(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate jaw tension indicators
        
        Args:
            landmarks: Full landmark array (468, 3)
        
        Returns:
            Dict with jaw measurements
        """
        jaw = self.get_jaw_landmarks(landmarks)
        
        # Distance between jaw landmarks
        left_jaw_distance = calculate_distance(jaw['left'][0], jaw['left'][1])
        right_jaw_distance = calculate_distance(jaw['right'][0], jaw['right'][1])
        
        # Width of jaw
        jaw_width = calculate_distance(jaw['left'][0], jaw['right'][0])
        
        return {
            'left_tension': left_jaw_distance,
            'right_tension': right_jaw_distance,
            'width': jaw_width,
            'average_tension': (left_jaw_distance + right_jaw_distance) / 2
        }
    
    def calculate_nasolabial_depth(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate nasolabial fold indicators
        
        Args:
            landmarks: Full landmark array (468, 3)
        
        Returns:
            Dict with nasolabial measurements
        """
        nasolabial = self.get_nasolabial_landmarks(landmarks)
        
        # Use z-depth as indicator of fold depth
        left_depth = nasolabial['left'][0][2]
        right_depth = nasolabial['right'][0][2]
        
        return {
            'left_depth': float(left_depth),
            'right_depth': float(right_depth),
            'average_depth': float((left_depth + right_depth) / 2),
            'asymmetry': float(abs(left_depth - right_depth))
        }
    
    def get_landmark_positions(self, landmarks: np.ndarray, indices: List[int]) -> np.ndarray:
        """
        Get positions of specific landmarks by indices
        
        Args:
            landmarks: Full landmark array (468, 3)
            indices: List of landmark indices
        
        Returns:
            np.ndarray: Selected landmark positions
        """
        return landmarks[indices]
    
    def normalize_to_face_size(self, value: float, landmarks: np.ndarray) -> float:
        """
        Normalize a measurement relative to face size
        Uses eye distance as reference
        
        Args:
            value: Value to normalize
            landmarks: Full landmark array (468, 3)
        
        Returns:
            float: Normalized value
        """
        # Use eye distance as reference for face size
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        eye_distance = calculate_distance(left_eye, right_eye)
        
        # Avoid division by zero
        if eye_distance < 0.001:
            return 0.0
        
        return value / eye_distance
