"""
Quality Validator
Validates face detection quality, lighting, and visibility
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    FACE_DETECTION_QUALITY,
    LIGHTING_QUALITY_THRESHOLD,
    CONFIDENCE_THRESHOLDS,
    ERROR_MESSAGES
)
from src.utils import check_lighting_quality


class QualityValidator:
    """
    Validates quality of face detection and image conditions
    """
    
    def __init__(self):
        """Initialize quality validator"""
        self.quality_thresholds = FACE_DETECTION_QUALITY
        self.confidence_thresholds = CONFIDENCE_THRESHOLDS
        self.lighting_threshold = LIGHTING_QUALITY_THRESHOLD
    
    def validate_image(self, image: np.ndarray) -> Tuple[bool, Dict]:
        """
        Validate image quality
        
        Args:
            image: Input image
        
        Returns:
            Tuple[bool, Dict]: (is_valid, validation_results)
        """
        results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'brightness': 0.0,
            'lighting_quality': 'Unknown',
            'resolution': (0, 0),
            'blur_score': 0.0
        }
        
        if image is None or image.size == 0:
            results['valid'] = False
            results['issues'].append("Invalid or empty image")
            return False, results
        
        # Check resolution
        height, width = image.shape[:2]
        results['resolution'] = (width, height)
        
        if width < 320 or height < 240:
            results['warnings'].append("Low resolution image (minimum 320x240 recommended)")
        
        # Check lighting
        brightness, lighting_quality = check_lighting_quality(image)
        results['brightness'] = brightness
        results['lighting_quality'] = lighting_quality
        
        if brightness < self.lighting_threshold:
            results['issues'].append(ERROR_MESSAGES['poor_lighting'])
        elif brightness < 100:
            results['warnings'].append("Suboptimal lighting conditions")
        
        # Check for blur (using Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        results['blur_score'] = float(blur_score)
        
        if blur_score < 100:
            results['warnings'].append("Image may be blurry")
        
        results['valid'] = len(results['issues']) == 0
        return results['valid'], results
    
    def validate_detection(self, metadata: Dict) -> Tuple[bool, Dict]:
        """
        Validate face detection results
        
        Args:
            metadata: Detection metadata from FaceDetector
        
        Returns:
            Tuple[bool, Dict]: (is_valid, validation_results)
        """
        results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'confidence_level': 'Unknown',
            'quality_level': 'Unknown',
            'reliability': 0.0
        }
        
        # Check if face was detected
        if not metadata.get('face_visible', False):
            results['valid'] = False
            results['issues'].append(ERROR_MESSAGES['no_face'])
            return False, results
        
        # Check number of faces
        num_faces = metadata.get('num_faces', 0)
        if num_faces == 0:
            results['valid'] = False
            results['issues'].append(ERROR_MESSAGES['no_face'])
            return False, results
        elif num_faces > 1:
            results['warnings'].append(ERROR_MESSAGES['multiple_faces'])
        
        # Check quality score
        quality_score = metadata.get('quality_score', 0.0)
        results['quality_level'] = self._get_quality_level(quality_score)
        
        if quality_score < self.quality_thresholds['fair']:
            results['issues'].append("Face detection quality too low")
        elif quality_score < self.quality_thresholds['good']:
            results['warnings'].append("Face detection quality is fair, results may vary")
        
        # Check detection confidence
        confidence = metadata.get('detection_confidence', 0.0)
        results['confidence_level'] = self._get_confidence_level(confidence)
        
        if confidence < self.confidence_thresholds['low']:
            results['issues'].append(ERROR_MESSAGES['low_confidence'])
        elif confidence < self.confidence_thresholds['medium']:
            results['warnings'].append("Moderate detection confidence")
        
        # Calculate overall reliability
        results['reliability'] = (quality_score + confidence) / 2
        
        # Add any warnings from metadata
        if 'warnings' in metadata:
            results['warnings'].extend(metadata['warnings'])
        
        results['valid'] = len(results['issues']) == 0
        return results['valid'], results
    
    def validate_landmarks(self, landmarks: np.ndarray) -> Tuple[bool, Dict]:
        """
        Validate landmark quality and completeness
        
        Args:
            landmarks: Landmark array (468, 3)
        
        Returns:
            Tuple[bool, Dict]: (is_valid, validation_results)
        """
        results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'completeness': 0.0,
            'stability': 0.0
        }
        
        if landmarks is None:
            results['valid'] = False
            results['issues'].append("No landmarks provided")
            return False, results
        
        # Check landmark count
        if len(landmarks) != 468:
            results['valid'] = False
            results['issues'].append(f"Invalid landmark count: {len(landmarks)} (expected 468)")
            return False, results
        
        # Check for NaN or infinite values
        if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
            results['valid'] = False
            results['issues'].append("Invalid landmark values detected")
            return False, results
        
        # Check if landmarks are within valid range [0, 1] for x and y
        x_in_range = np.all((landmarks[:, 0] >= -0.1) & (landmarks[:, 0] <= 1.1))
        y_in_range = np.all((landmarks[:, 1] >= -0.1) & (landmarks[:, 1] <= 1.1))
        
        if not (x_in_range and y_in_range):
            results['warnings'].append("Some landmarks are outside expected range")
        
        # Calculate completeness (percentage of landmarks within normal range)
        valid_x = np.sum((landmarks[:, 0] >= 0) & (landmarks[:, 0] <= 1))
        valid_y = np.sum((landmarks[:, 1] >= 0) & (landmarks[:, 1] <= 1))
        results['completeness'] = float(min(valid_x, valid_y) / len(landmarks))
        
        if results['completeness'] < 0.90:
            results['warnings'].append(f"Only {results['completeness']*100:.1f}% of landmarks are valid")
        
        # Calculate stability (low z-variance indicates stable detection)
        z_variance = np.var(landmarks[:, 2])
        results['stability'] = float(max(0, 1.0 - z_variance * 10))
        
        if results['stability'] < 0.6:
            results['warnings'].append("Landmark detection may be unstable")
        
        results['valid'] = len(results['issues']) == 0
        return results['valid'], results
    
    def validate_face_visibility(self, landmarks: np.ndarray) -> Tuple[bool, Dict]:
        """
        Validate that face is fully visible and properly positioned
        
        Args:
            landmarks: Landmark array (468, 3)
        
        Returns:
            Tuple[bool, Dict]: (is_valid, validation_results)
        """
        results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'centering': 0.0,
            'size_score': 0.0,
            'rotation': 0.0
        }
        
        # Check face position
        min_x, max_x = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
        min_y, max_y = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
        
        # Check if face is too close to edges
        margin = 0.02
        if min_x < margin or max_x > (1 - margin):
            results['issues'].append("Face too close to left/right edge")
        
        if min_y < margin or max_y > (1 - margin):
            results['issues'].append("Face too close to top/bottom edge")
        
        # Calculate centering
        face_center_x = (min_x + max_x) / 2
        face_center_y = (min_y + max_y) / 2
        
        center_offset_x = abs(face_center_x - 0.5)
        center_offset_y = abs(face_center_y - 0.5)
        results['centering'] = 1.0 - ((center_offset_x + center_offset_y) / 2)
        
        if results['centering'] < 0.6:
            results['warnings'].append("Face is off-center")
        
        # Check face size
        face_width = max_x - min_x
        face_height = max_y - min_y
        
        if face_width < 0.3 or face_height < 0.4:
            results['issues'].append("Face too small (move closer to camera)")
            results['size_score'] = 0.3
        elif face_width > 0.8 or face_height > 0.9:
            results['issues'].append("Face too large (move away from camera)")
            results['size_score'] = 0.3
        else:
            results['size_score'] = 1.0
        
        # Check face rotation (frontal vs profile)
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_offset = abs(nose_tip[0] - eye_center_x)
        results['rotation'] = float(nose_offset)
        
        if nose_offset > 0.08:
            results['issues'].append("Face not frontal (turn to face camera)")
        elif nose_offset > 0.05:
            results['warnings'].append("Face slightly rotated")
        
        results['valid'] = len(results['issues']) == 0
        return results['valid'], results
    
    def validate_complete(self, 
                         image: np.ndarray, 
                         landmarks: np.ndarray, 
                         metadata: Dict) -> Tuple[bool, Dict]:
        """
        Perform complete validation of image, detection, and landmarks
        
        Args:
            image: Input image
            landmarks: Landmark array
            metadata: Detection metadata
        
        Returns:
            Tuple[bool, Dict]: (is_valid, comprehensive_results)
        """
        comprehensive_results = {
            'overall_valid': True,
            'image_validation': {},
            'detection_validation': {},
            'landmark_validation': {},
            'visibility_validation': {},
            'all_issues': [],
            'all_warnings': [],
            'recommendation': ''
        }
        
        # Validate image
        image_valid, image_results = self.validate_image(image)
        comprehensive_results['image_validation'] = image_results
        
        # Validate detection
        detection_valid, detection_results = self.validate_detection(metadata)
        comprehensive_results['detection_validation'] = detection_results
        
        # Validate landmarks (only if detection was successful)
        if landmarks is not None:
            landmark_valid, landmark_results = self.validate_landmarks(landmarks)
            comprehensive_results['landmark_validation'] = landmark_results
            
            # Validate visibility
            visibility_valid, visibility_results = self.validate_face_visibility(landmarks)
            comprehensive_results['visibility_validation'] = visibility_results
        else:
            landmark_valid = False
            visibility_valid = False
        
        # Collect all issues and warnings
        for key in ['image_validation', 'detection_validation', 'landmark_validation', 'visibility_validation']:
            if key in comprehensive_results and comprehensive_results[key]:
                comprehensive_results['all_issues'].extend(comprehensive_results[key].get('issues', []))
                comprehensive_results['all_warnings'].extend(comprehensive_results[key].get('warnings', []))
        
        # Determine overall validity
        comprehensive_results['overall_valid'] = (
            image_valid and 
            detection_valid and 
            landmark_valid and 
            visibility_valid
        )
        
        # Generate recommendation
        if comprehensive_results['overall_valid']:
            comprehensive_results['recommendation'] = "Conditions are optimal for pain assessment"
        elif not detection_valid:
            comprehensive_results['recommendation'] = "Ensure face is visible and well-lit"
        elif not visibility_valid:
            comprehensive_results['recommendation'] = "Adjust position to face camera directly"
        else:
            comprehensive_results['recommendation'] = "Improve lighting and image quality"
        
        return comprehensive_results['overall_valid'], comprehensive_results
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level from score"""
        if score >= self.quality_thresholds['excellent']:
            return "Excellent"
        elif score >= self.quality_thresholds['good']:
            return "Good"
        elif score >= self.quality_thresholds['fair']:
            return "Fair"
        else:
            return "Poor"
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level from score"""
        if score >= self.confidence_thresholds['high']:
            return "High"
        elif score >= self.confidence_thresholds['medium']:
            return "Medium"
        else:
            return "Low"
