"""
Face Detector using MediaPipe Face Mesh
Handles face detection, landmark extraction, and quality validation
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List, Dict
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import MEDIAPIPE_CONFIG, FACE_DETECTION_QUALITY, LIGHTING_QUALITY_THRESHOLD
from src.logger import setup_logger, get_logger
from src.utils import check_lighting_quality


class FaceDetector:
    """
    MediaPipe Face Mesh wrapper for detecting faces and extracting landmarks
    """
    
    def __init__(self, 
                 max_num_faces: int = None,
                 min_detection_confidence: float = None,
                 min_tracking_confidence: float = None,
                 refine_landmarks: bool = None):
        """
        Initialize Face Detector with MediaPipe Face Mesh
        
        Args:
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            refine_landmarks: Whether to refine landmarks around eyes and lips
        """
        self.logger = setup_logger(__name__)
        
        # Use config defaults if not specified
        config = MEDIAPIPE_CONFIG
        self.max_num_faces = max_num_faces or config['max_num_faces']
        self.min_detection_confidence = min_detection_confidence or config['min_detection_confidence']
        self.min_tracking_confidence = min_tracking_confidence or config['min_tracking_confidence']
        self.refine_landmarks = refine_landmarks if refine_landmarks is not None else config['refine_landmarks']
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # For drawing landmarks
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.logger.info("FaceDetector initialized successfully")
    
    def detect(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Dict]:
        """
        Detect face and extract landmarks from image
        
        Args:
            image: Input image (BGR format from OpenCV)
        
        Returns:
            Tuple containing:
                - success (bool): Whether face was detected
                - landmarks (np.ndarray): Array of shape (468, 3) with x, y, z coordinates
                - metadata (dict): Additional information about detection
        """
        metadata = {
            'num_faces': 0,
            'detection_confidence': 0.0,
            'face_visible': False,
            'quality_score': 0.0,
            'lighting_score': 0.0,
            'lighting_quality': 'Unknown',
            'warnings': []
        }
        
        if image is None or image.size == 0:
            self.logger.warning("Empty or invalid image provided")
            metadata['warnings'].append("Invalid image")
            return False, None, metadata
        
        # Check lighting quality
        brightness, lighting_quality = check_lighting_quality(image)
        metadata['lighting_score'] = float(brightness)
        metadata['lighting_quality'] = lighting_quality
        
        if brightness < LIGHTING_QUALITY_THRESHOLD:
            metadata['warnings'].append(f"Poor lighting detected ({lighting_quality})")
            self.logger.warning(f"Poor lighting: {brightness:.1f}")
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_mesh.process(image_rgb)
        
        # Check if faces were detected
        if not results.multi_face_landmarks:
            self.logger.debug("No faces detected in image")
            return False, None, metadata
        
        # Get number of faces
        num_faces = len(results.multi_face_landmarks)
        metadata['num_faces'] = num_faces
        
        if num_faces > 1:
            metadata['warnings'].append(f"Multiple faces detected ({num_faces})")
            self.logger.warning(f"Multiple faces detected: {num_faces}")
        
        # Extract landmarks from first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to numpy array
        landmarks = self._extract_landmark_array(face_landmarks, image.shape)
        
        # Calculate detection quality
        quality_score = self._calculate_quality_score(landmarks, image.shape)
        metadata['quality_score'] = quality_score
        metadata['face_visible'] = True
        
        # Estimate confidence (MediaPipe doesn't directly provide this)
        metadata['detection_confidence'] = quality_score
        
        self.logger.debug(f"Face detected successfully. Quality: {quality_score:.2f}")
        
        return True, landmarks, metadata
    
    def _extract_landmark_array(self, face_landmarks, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract landmark coordinates as numpy array
        
        Args:
            face_landmarks: MediaPipe face landmarks object
            image_shape: Shape of the image (height, width, channels)
        
        Returns:
            np.ndarray: Array of shape (468, 3) with normalized coordinates
        """
        height, width = image_shape[:2]
        
        landmarks = np.zeros((len(face_landmarks.landmark), 3), dtype=np.float32)
        
        for idx, landmark in enumerate(face_landmarks.landmark):
            landmarks[idx] = [landmark.x, landmark.y, landmark.z]
        
        return landmarks
    
    def _calculate_quality_score(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """
        Calculate face detection quality score
        
        Args:
            landmarks: Landmark array (468, 3)
            image_shape: Image shape (height, width, channels)
        
        Returns:
            float: Quality score between 0 and 1
        """
        height, width = image_shape[:2]
        
        # Check if face is centered and well-positioned
        face_center_x = np.mean(landmarks[:, 0])
        face_center_y = np.mean(landmarks[:, 1])
        
        # Calculate how centered the face is (0 = edge, 1 = center)
        center_x = abs(face_center_x - 0.5) * 2  # 0 at center, 1 at edge
        center_y = abs(face_center_y - 0.5) * 2
        centering_score = 1.0 - ((center_x + center_y) / 2)
        
        # Check face size (using distance between eyes)
        left_eye = landmarks[33]  # Left eye landmark
        right_eye = landmarks[263]  # Right eye landmark
        eye_distance = np.linalg.norm(left_eye[:2] - right_eye[:2])
        
        # Good face size is when eye distance is 15-35% of image width
        size_score = 1.0 if 0.15 <= eye_distance <= 0.35 else 0.5
        
        # Check landmark confidence (using z-depth variance)
        # Less variance in z means more stable detection
        z_variance = np.var(landmarks[:, 2])
        depth_score = max(0.5, 1.0 - z_variance * 10)
        
        # Check if face is too close to edges
        min_x, max_x = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
        min_y, max_y = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
        
        edge_margin = 0.05  # 5% margin
        edge_score = 1.0
        if min_x < edge_margin or max_x > (1 - edge_margin):
            edge_score *= 0.7
        if min_y < edge_margin or max_y > (1 - edge_margin):
            edge_score *= 0.7
        
        # Combined quality score
        quality_score = (
            centering_score * 0.3 +
            size_score * 0.3 +
            depth_score * 0.2 +
            edge_score * 0.2
        )
        
        return float(np.clip(quality_score, 0.0, 1.0))
    
    def get_quality_level(self, quality_score: float) -> str:
        """
        Get quality level description from score
        
        Args:
            quality_score: Quality score (0-1)
        
        Returns:
            str: Quality level description
        """
        if quality_score >= FACE_DETECTION_QUALITY['excellent']:
            return "Excellent"
        elif quality_score >= FACE_DETECTION_QUALITY['good']:
            return "Good"
        elif quality_score >= FACE_DETECTION_QUALITY['fair']:
            return "Fair"
        else:
            return "Poor"
    
    def draw_landmarks(self, 
                       image: np.ndarray, 
                       landmarks: np.ndarray,
                       draw_connections: bool = True) -> np.ndarray:
        """
        Draw face mesh landmarks on image
        
        Args:
            image: Input image
            landmarks: Landmark array (468, 3)
            draw_connections: Whether to draw connections between landmarks
        
        Returns:
            np.ndarray: Image with drawn landmarks
        """
        annotated_image = image.copy()
        height, width = image.shape[:2]
        
        # Convert normalized landmarks to pixel coordinates
        for idx, landmark in enumerate(landmarks):
            x = int(landmark[0] * width)
            y = int(landmark[1] * height)
            
            # Draw landmark point
            cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)
        
        return annotated_image
    
    def draw_landmarks_mediapipe(self, image: np.ndarray) -> np.ndarray:
        """
        Draw face mesh using MediaPipe's built-in drawing utilities
        
        Args:
            image: Input image (BGR)
        
        Returns:
            np.ndarray: Image with drawn face mesh
        """
        annotated_image = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        
        return annotated_image
    
    def get_specific_landmarks(self, landmarks: np.ndarray, indices: List[int]) -> np.ndarray:
        """
        Extract specific landmarks by indices
        
        Args:
            landmarks: Full landmark array (468, 3)
            indices: List of landmark indices to extract
        
        Returns:
            np.ndarray: Selected landmarks
        """
        return landmarks[indices]
    
    def validate_face_visibility(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[bool, List[str]]:
        """
        Validate that face is fully visible and not occluded
        
        Args:
            landmarks: Landmark array (468, 3)
            image_shape: Image shape (height, width, channels)
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of issues)
        """
        issues = []
        
        # Check if key landmarks are within image bounds
        min_x, max_x = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
        min_y, max_y = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
        
        margin = 0.02  # 2% margin
        if min_x < margin or max_x > (1 - margin):
            issues.append("Face too close to left/right edge")
        
        if min_y < margin or max_y > (1 - margin):
            issues.append("Face too close to top/bottom edge")
        
        # Check face orientation (using nose and eye landmarks)
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        # Calculate face rotation
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_offset = abs(nose_tip[0] - eye_center_x)
        
        if nose_offset > 0.05:  # Face is rotated more than 5%
            issues.append("Face not frontal (head turned)")
        
        # Check if face is too far or too close
        eye_distance = np.linalg.norm(left_eye[:2] - right_eye[:2])
        if eye_distance < 0.10:
            issues.append("Face too far from camera")
        elif eye_distance > 0.45:
            issues.append("Face too close to camera")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def close(self):
        """Release resources"""
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except:
                pass  # Ignore errors during cleanup
        if hasattr(self, 'logger'):
            self.logger.info("FaceDetector closed")
    
    def __del__(self):
        """Destructor"""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup
