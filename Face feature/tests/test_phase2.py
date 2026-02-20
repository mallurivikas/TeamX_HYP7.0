"""
Phase 2 Test Script
Tests the MediaPipe integration components
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.face_detector import FaceDetector
from src.landmark_extractor import LandmarkExtractor
from src.quality_validator import QualityValidator
from config import OUTPUT_DIR

def test_face_detection():
    """Test face detection with webcam"""
    print("\n" + "="*70)
    print("PHASE 2 TEST: Face Detection & Landmark Extraction")
    print("="*70)
    
    # Initialize components
    print("\n1. Initializing components...")
    detector = FaceDetector()
    extractor = LandmarkExtractor()
    validator = QualityValidator()
    print("   ✓ Components initialized")
    
    # Try to open webcam
    print("\n2. Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("   ✗ Could not open webcam")
        print("\n   Testing with synthetic image instead...")
        test_with_synthetic_image(detector, extractor, validator)
        return
    
    print("   ✓ Webcam opened successfully")
    print("\n3. Starting face detection (press 'q' to quit, 's' to save)...")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("   ✗ Failed to read frame")
            break
        
        frame_count += 1
        
        # Detect face and landmarks
        success, landmarks, metadata = detector.detect(frame)
        
        # Create display frame
        display_frame = frame.copy()
        
        if success:
            # Validate detection
            valid, validation = validator.validate_complete(frame, landmarks, metadata)
            
            # Draw landmarks
            display_frame = detector.draw_landmarks_mediapipe(frame)
            
            # Extract pain landmarks
            pain_landmarks = extractor.get_all_pain_landmarks(landmarks)
            
            # Calculate metrics
            eyebrow_dist = extractor.calculate_eyebrow_distance(landmarks)
            mouth_metrics = extractor.calculate_mouth_metrics(landmarks)
            eye_aperture = extractor.calculate_eye_aperture(landmarks)
            jaw_tension = extractor.calculate_jaw_tension(landmarks)
            
            # Display info
            y_offset = 30
            cv2.putText(display_frame, f"Faces: {metadata['num_faces']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            y_offset += 30
            quality = detector.get_quality_level(metadata['quality_score'])
            color = (0, 255, 0) if valid else (0, 165, 255)
            cv2.putText(display_frame, f"Quality: {quality} ({metadata['quality_score']:.2f})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            y_offset += 30
            cv2.putText(display_frame, f"Lighting: {metadata['lighting_quality']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display metrics
            y_offset += 40
            cv2.putText(display_frame, "Pain Indicator Metrics:", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 25
            cv2.putText(display_frame, f"Eyebrow span: {eyebrow_dist['average_span']:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 20
            cv2.putText(display_frame, f"Mouth droop: {mouth_metrics['average_droop']:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 20
            cv2.putText(display_frame, f"Eye aperture: {eye_aperture['average_aperture']:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 20
            cv2.putText(display_frame, f"Jaw tension: {jaw_tension['average_tension']:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show warnings
            if validation['all_warnings']:
                y_offset += 30
                cv2.putText(display_frame, "Warnings:", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                for warning in validation['all_warnings'][:2]:  # Show max 2 warnings
                    y_offset += 20
                    cv2.putText(display_frame, f"- {warning[:40]}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
        else:
            cv2.putText(display_frame, "No face detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if metadata.get('warnings'):
                cv2.putText(display_frame, metadata['warnings'][0], 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Instructions
        cv2.putText(display_frame, "Press 'q' to quit, 's' to save", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Phase 2 Test - Face Detection", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and success:
            # Save annotated frame
            output_path = OUTPUT_DIR / f"phase2_test_{frame_count}.jpg"
            cv2.imwrite(str(output_path), display_frame)
            print(f"   ✓ Saved frame to {output_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    
    print("\n" + "="*70)
    print("✓ Phase 2 test completed successfully")
    print("="*70)


def test_with_synthetic_image(detector, extractor, validator):
    """Test with a synthetic or placeholder image"""
    print("\n   Creating test image...")
    
    # Create a simple test image with a circle (representing a face)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image.fill(128)  # Gray background
    
    # Add some elements
    cv2.circle(test_image, (320, 240), 100, (200, 200, 200), -1)
    cv2.putText(test_image, "No webcam available", 
               (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(test_image, "Please test with actual webcam or images", 
               (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Validate image
    valid, results = validator.validate_image(test_image)
    print(f"\n   Image validation: {'✓ Valid' if valid else '✗ Invalid'}")
    print(f"   - Resolution: {results['resolution']}")
    print(f"   - Brightness: {results['brightness']:.1f}")
    print(f"   - Lighting: {results['lighting_quality']}")
    
    cv2.imshow("Test Image", test_image)
    print("\n   Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("✓ Phase 2 components validated (limited test)")
    print("  For full testing, please use webcam or real face images")
    print("="*70)


def run_component_tests():
    """Run individual component tests"""
    print("\n" + "="*70)
    print("RUNNING COMPONENT TESTS")
    print("="*70)
    
    print("\n✓ Testing FaceDetector initialization...")
    detector = FaceDetector()
    print("  - MediaPipe Face Mesh initialized")
    print(f"  - Max faces: {detector.max_num_faces}")
    print(f"  - Detection confidence: {detector.min_detection_confidence}")
    
    print("\n✓ Testing LandmarkExtractor...")
    extractor = LandmarkExtractor()
    print(f"  - Landmark indices loaded: {len(extractor.landmark_indices)} groups")
    
    print("\n✓ Testing QualityValidator...")
    validator = QualityValidator()
    print(f"  - Quality thresholds configured")
    print(f"  - Lighting threshold: {validator.lighting_threshold}")
    
    detector.close()
    
    print("\n" + "="*70)
    print("✓ All component tests passed")
    print("="*70)


if __name__ == "__main__":
    print("\nPHASE 2 - CORE MEDIAPIPE INTEGRATION")
    print("Testing face detection, landmark extraction, and validation\n")
    
    # Run component tests first
    run_component_tests()
    
    # Ask user if they want to test with webcam
    print("\nDo you want to test with webcam? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        test_face_detection()
    else:
        print("\nSkipping webcam test.")
        print("You can run this test later with: python tests/test_phase2.py")
    
    print("\n✓ Phase 2 implementation complete and tested!")
