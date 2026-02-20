"""
Facial Expression Analysis Agent
Integrates Face Feature module for real-time pain/stress detection
"""
import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add Face feature module to path
current_dir = Path(__file__).parent.parent
face_feature_path = current_dir / 'Face feature' / 'src'
sys.path.insert(0, str(face_feature_path))

try:
    import mediapipe as mp
    from pain_analyzers.brow_analyzer import BrowAnalyzer
    from pain_analyzers.eye_analyzer import EyeAnalyzer
    from pain_analyzers.grimace_analyzer import GrimaceAnalyzer
    from pain_analyzers.jaw_analyzer import JawAnalyzer
    from pain_analyzers.nasolabial_analyzer import NasolabialAnalyzer
    FACE_FEATURE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Face feature module not available: {e}")
    FACE_FEATURE_AVAILABLE = False


class FacialAgent:
    """
    Real-time facial expression analyzer for health assessment
    
    Detects:
    - Pain indicators (grimacing, jaw tension, brow furrowing)
    - Stress markers (eye squinting, facial tension)
    - Anxiety indicators (combined facial features)
    """
    
    def __init__(self):
        if not FACE_FEATURE_AVAILABLE:
            self.available = False
            return
            
        self.available = True
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices (from Face feature config)
        self.landmark_indices = self._get_landmark_indices()
        
        # Initialize pain analyzers
        self.analyzers = {
            'brow': BrowAnalyzer(self.landmark_indices),
            'eye': EyeAnalyzer(self.landmark_indices),
            'grimace': GrimaceAnalyzer(self.landmark_indices),
            'jaw': JawAnalyzer(self.landmark_indices),
            'nasolabial': NasolabialAnalyzer(self.landmark_indices)
        }
        
        self.baseline_set = False
        self.frame_count = 0
        self.pain_scores = []
        
    def _get_landmark_indices(self):
        """Get landmark indices for analyzers"""
        return {
            'brow': {'left': [70, 63, 105, 66, 107], 'right': [336, 296, 334, 293, 300]},
            'eye': {'left': [33, 160, 158, 133, 153, 144], 'right': [362, 385, 387, 263, 373, 380]},
            'mouth': [61, 291, 0, 17, 269, 405, 314, 17, 84, 181, 91, 146],
            'jaw': [234, 454, 323, 93],
            'nasolabial': [48, 278, 36, 266]
        }
    
    def process_frame(self, frame):
        """
        Process a single video frame
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            dict with:
            - success: bool
            - pain_score: float (0-10)
            - stress_score: float (0-10)
            - anxiety_score: float (0-10)
            - annotated_frame: frame with face mesh overlay
        """
        if not self.available:
            return {
                'success': False,
                'error': 'Facial analysis not available',
                'annotated_frame': frame
            }
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return {
                'success': False,
                'error': 'No face detected',
                'annotated_frame': frame
            }
        
        # Extract landmarks
        landmarks = self._extract_landmarks(results.multi_face_landmarks[0], frame.shape)
        face_landmarks = results.multi_face_landmarks[0]  # Store for later use
        
        # Set baseline on first detection
        if not self.baseline_set and self.frame_count == 0:
            self._set_baseline(landmarks)
        
        # Analyze pain indicators
        analysis = self._analyze_pain(landmarks)
        
        # Draw face mesh overlay
        annotated_frame = self._draw_overlay(frame.copy(), results)
        
        # Store pain score
        self.pain_scores.append(analysis['pain_score'])
        self.frame_count += 1
        
        # Extract landmark coordinates for frontend rendering
        landmarks_list = []
        for lm in face_landmarks.landmark:
            landmarks_list.append({
                'x': float(lm.x),
                'y': float(lm.y),
                'z': float(lm.z)
            })
        
        return {
            'success': True,
            'pain_score': analysis['pain_score'],
            'stress_score': analysis['stress_score'],
            'anxiety_score': analysis['anxiety_score'],
            'indicators': analysis['indicators'],
            'annotated_frame': annotated_frame,
            'face_mesh': {
                'landmarks': landmarks_list
            }
        }
    
    def _extract_landmarks(self, face_landmarks, image_shape):
        """Convert MediaPipe landmarks to numpy array"""
        h, w = image_shape[:2]
        landmarks = []
        for lm in face_landmarks.landmark:
            landmarks.append([lm.x * w, lm.y * h, lm.z * w])
        return np.array(landmarks)
    
    def _set_baseline(self, landmarks):
        """Set neutral baseline for all analyzers"""
        for analyzer in self.analyzers.values():
            analyzer.set_baseline(landmarks)
        self.baseline_set = True
    
    def _analyze_pain(self, landmarks):
        """Analyze all pain indicators"""
        results = {}
        for name, analyzer in self.analyzers.items():
            results[name] = analyzer.analyze(landmarks)
        
        # Calculate weighted scores
        weights = {
            'brow': 0.25,
            'grimace': 0.30,
            'eye': 0.20,
            'jaw': 0.20,
            'nasolabial': 0.05
        }
        
        pain_score = sum(
            results[name]['smoothed_score'] * weights[name]
            for name in weights.keys()
        ) * 10  # Scale to 0-10
        
        stress_score = (
            results['eye']['smoothed_score'] * 0.50 +
            results['brow']['smoothed_score'] * 0.30 +
            results['jaw']['smoothed_score'] * 0.20
        ) * 10
        
        anxiety_score = (
            results['eye']['smoothed_score'] * 0.30 +
            results['brow']['smoothed_score'] * 0.30 +
            results['jaw']['smoothed_score'] * 0.25 +
            results['grimace']['smoothed_score'] * 0.15
        ) * 10
        
        return {
            'pain_score': pain_score,
            'stress_score': stress_score,
            'anxiety_score': anxiety_score,
            'indicators': {
                name: result['smoothed_percentage']
                for name, result in results.items()
            }
        }
    
    def _draw_overlay(self, frame, results):
        """Draw face mesh overlay on frame"""
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        for face_landmarks in results.multi_face_landmarks:
            # Draw tesselation (light overlay)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # Draw contours (visible landmarks)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        return frame
    
    def get_average_scores(self):
        """Get average scores from all recorded frames"""
        if not self.pain_scores:
            return {
                'avg_pain_score': 0.0,
                'avg_stress_score': 0.0,
                'avg_anxiety_score': 0.0,
                'total_frames': 0
            }
        
        return {
            'avg_pain_score': np.mean(self.pain_scores),
            'avg_stress_score': np.mean([s for s in self.pain_scores]),  # Placeholder
            'avg_anxiety_score': np.mean([s for s in self.pain_scores]),  # Placeholder
            'total_frames': self.frame_count
        }
    
    def reset(self):
        """Reset analyzer state"""
        self.baseline_set = False
        self.frame_count = 0
        self.pain_scores = []
        for analyzer in self.analyzers.values():
            analyzer.reset_history()
    
    def cleanup(self):
        """Release resources"""
        if self.available:
            self.face_mesh.close()


# Convenience function for one-shot analysis
def analyze_facial_image(image_path):
    """
    Analyze a single image for facial pain indicators
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict with pain/stress/anxiety scores
    """
    agent = FacialAgent()
    
    if not agent.available:
        return {'error': 'Facial analysis not available'}
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        return {'error': 'Could not read image'}
    
    # Process
    result = agent.process_frame(frame)
    agent.cleanup()
    
    return result
