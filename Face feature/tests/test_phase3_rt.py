"""
Real-time Pain Detection Test
Tests Phase 3 analyzers with live webcam feed
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pain_analyzers import (
    BrowAnalyzer,
    GrimaceAnalyzer,
    EyeAnalyzer,
    JawAnalyzer,
    NasolabialAnalyzer
)
from config import LANDMARK_INDICES

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class RealtimePainDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize all analyzers with landmark indices from config
        self.brow_analyzer = BrowAnalyzer(LANDMARK_INDICES)
        self.grimace_analyzer = GrimaceAnalyzer(LANDMARK_INDICES)
        self.eye_analyzer = EyeAnalyzer(LANDMARK_INDICES)
        self.jaw_analyzer = JawAnalyzer(LANDMARK_INDICES)
        self.nasolabial_analyzer = NasolabialAnalyzer(LANDMARK_INDICES)
        
        self.baseline_set = False
        
    def extract_landmarks(self, results, image_shape):
        """Extract landmarks as numpy array"""
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        h, w = image_shape[:2]
        
        landmark_array = []
        for lm in landmarks.landmark:
            landmark_array.append([lm.x * w, lm.y * h, lm.z * w])
            
        return np.array(landmark_array)
    
    def calibrate_baseline(self, landmarks):
        """Set baseline for all analyzers"""
        self.brow_analyzer.set_baseline(landmarks)
        self.grimace_analyzer.set_baseline(landmarks)
        self.eye_analyzer.set_baseline(landmarks)
        self.jaw_analyzer.set_baseline(landmarks)
        self.nasolabial_analyzer.set_baseline(landmarks)
        self.baseline_set = True
        print("✓ Baseline calibrated!")
    
    def reset_baseline(self):
        """Reset baseline for all analyzers"""
        self.brow_analyzer.reset_history()
        self.grimace_analyzer.reset_history()
        self.eye_analyzer.reset_history()
        self.jaw_analyzer.reset_history()
        self.nasolabial_analyzer.reset_history()
        self.baseline_set = False
        print("✓ Baseline reset!")
    
    def analyze_pain(self, landmarks):
        """Analyze pain indicators"""
        brow_result = self.brow_analyzer.analyze(landmarks)
        grimace_result = self.grimace_analyzer.analyze(landmarks)
        eye_result = self.eye_analyzer.analyze(landmarks)
        jaw_result = self.jaw_analyzer.analyze(landmarks)
        nasolabial_result = self.nasolabial_analyzer.analyze(landmarks)
        
        # Calculate weighted pain score (0-10)
        weights = {
            'brow': 0.25,
            'grimace': 0.30,
            'eye': 0.20,
            'jaw': 0.20,
            'nasolabial': 0.05
        }
        
        overall_score = (
            brow_result['smoothed_score'] * weights['brow'] +
            grimace_result['smoothed_score'] * weights['grimace'] +
            eye_result['smoothed_score'] * weights['eye'] +
            jaw_result['smoothed_score'] * weights['jaw'] +
            nasolabial_result['smoothed_score'] * weights['nasolabial']
        ) * 10
        
        return {
            'brow': {'percentage': brow_result['smoothed_percentage']},
            'grimace': {'percentage': grimace_result['smoothed_percentage']},
            'eye': {'percentage': eye_result['smoothed_percentage']},
            'jaw': {'percentage': jaw_result['smoothed_percentage']},
            'nasolabial': {'percentage': nasolabial_result['smoothed_percentage']},
            'overall_score': overall_score
        }
    
    def draw_info(self, image, results_dict):
        """Draw pain scores on image"""
        h, w = image.shape[:2]
        
        # Background for text
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        y_offset = 40
        
        # Title
        cv2.putText(image, "PAIN DETECTION SYSTEM", (20, y_offset),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 35
        
        # Baseline status
        status_color = (0, 255, 0) if self.baseline_set else (0, 165, 255)
        status_text = "CALIBRATED" if self.baseline_set else "NOT CALIBRATED"
        cv2.putText(image, f"Baseline: {status_text}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y_offset += 30
        
        if results_dict:
            # Individual scores
            indicators = [
                ('Brow Tension', results_dict['brow']['percentage'], (0, 255, 255)),
                ('Grimace', results_dict['grimace']['percentage'], (0, 255, 255)),
                ('Eye Squint', results_dict['eye']['percentage'], (0, 255, 255)),
                ('Jaw Clench', results_dict['jaw']['percentage'], (0, 255, 255)),
                ('Nasolabial', results_dict['nasolabial']['percentage'], (0, 255, 255))
            ]
            
            for name, percentage, color in indicators:
                cv2.putText(image, f"{name}: {percentage:.1f}%", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25
            
            y_offset += 10
            
            # Overall pain score with color coding
            score = results_dict['overall_score']
            if score < 3:
                color = (0, 255, 0)  # Green
                level = "LOW"
            elif score < 6:
                color = (0, 255, 255)  # Yellow
                level = "MODERATE"
            else:
                color = (0, 0, 255)  # Red
                level = "HIGH"
            
            cv2.putText(image, f"Pain Score: {score:.1f}/10", (20, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.putText(image, f"[{level}]", (20, y_offset + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Controls
        cv2.putText(image, "Controls: C=Calibrate | R=Reset | S=Save | Q=Quit",
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open webcam")
            return
        
        print("\n" + "="*70)
        print("REAL-TIME PAIN DETECTION TEST - PHASE 3")
        print("="*70)
        print("\nControls:")
        print("  C - Calibrate baseline (capture neutral expression)")
        print("  R - Reset baseline")
        print("  S - Save current frame analysis")
        print("  Q - Quit")
        print("\n" + "="*70 + "\n")
        
        frame_count = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("❌ Failed to capture frame")
                continue
            
            # Flip image for selfie view
            image = cv2.flip(image, 1)
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process face mesh
            results = self.face_mesh.process(image_rgb)
            
            results_dict = None
            
            if results.multi_face_landmarks:
                # Draw face mesh
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                
                # Extract landmarks
                landmarks = self.extract_landmarks(results, image.shape)
                
                if landmarks is not None:
                    # Analyze pain
                    results_dict = self.analyze_pain(landmarks)
            
            # Draw info overlay
            self.draw_info(image, results_dict)
            
            # Show image
            cv2.imshow('Pain Detection - Phase 3 Test', image)
            
            # Handle key presses
            key = cv2.waitKey(5) & 0xFF
            
            if key == ord('q'):
                print("\n✓ Exiting...")
                break
            elif key == ord('c'):
                if results.multi_face_landmarks:
                    landmarks = self.extract_landmarks(results, image.shape)
                    if landmarks is not None:
                        self.calibrate_baseline(landmarks)
                else:
                    print("⚠ No face detected! Cannot calibrate.")
            elif key == ord('r'):
                self.reset_baseline()
            elif key == ord('s'):
                if results_dict:
                    filename = f"outputs/pain_analysis_{frame_count}.jpg"
                    cv2.imwrite(filename, image)
                    print(f"✓ Saved: {filename}")
                    print(f"  Pain Score: {results_dict['overall_score']:.1f}/10")
                else:
                    print("⚠ No face detected! Cannot save.")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()

if __name__ == "__main__":
    detector = RealtimePainDetector()
    detector.run()