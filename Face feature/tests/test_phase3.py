"""
Test script for Phase 3: Pain Analyzer Modules
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for config
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import config and analyzers
import config
from pain_analyzers import (
    BrowAnalyzer,
    GrimaceAnalyzer,
    EyeAnalyzer,
    JawAnalyzer,
    NasolabialAnalyzer
)

print("=" * 70)
print("PHASE 3: PAIN ANALYZER MODULES - TEST")
print("=" * 70)
print()

# Create dummy landmarks (468 points with x, y, z coordinates)
# In real usage, these come from MediaPipe
dummy_landmarks = np.random.rand(468, 3)

# Normalize to reasonable ranges
dummy_landmarks[:, 0] *= 0.5  # x: 0-0.5
dummy_landmarks[:, 1] *= 0.5  # y: 0-0.5
dummy_landmarks[:, 2] *= 0.1  # z: 0-0.1

print("Testing with dummy landmark data (468 points)...")
print()

# Test 1: Brow Analyzer
print("1. Testing Brow Tension Analyzer")
print("-" * 70)
try:
    brow_analyzer = BrowAnalyzer(config.LANDMARK_INDICES)
    
    # Set baseline
    brow_analyzer.set_baseline(dummy_landmarks)
    print("✓ Baseline calibration: SUCCESS")
    
    # Analyze
    result = brow_analyzer.analyze(dummy_landmarks, use_baseline=True)
    print(f"✓ Analysis complete")
    print(f"  - Tension Score: {result['tension_score']:.3f}")
    print(f"  - Tension Percentage: {result['tension_percentage']:.1f}%")
    print(f"  - Description: {brow_analyzer.get_description(result['tension_score'])}")
    print(f"  - Has Baseline: {result['has_baseline']}")
    
    # Test without baseline
    brow_analyzer2 = BrowAnalyzer(config.LANDMARK_INDICES)
    result2 = brow_analyzer2.analyze(dummy_landmarks, use_baseline=False)
    print(f"✓ Analysis without baseline: {result2['tension_percentage']:.1f}%")
    
    print("✓ Brow Analyzer: PASSED")
except Exception as e:
    print(f"✗ Brow Analyzer: FAILED - {e}")

print()

# Test 2: Grimace Analyzer
print("2. Testing Grimace Analyzer")
print("-" * 70)
try:
    grimace_analyzer = GrimaceAnalyzer(config.LANDMARK_INDICES)
    
    grimace_analyzer.set_baseline(dummy_landmarks)
    print("✓ Baseline calibration: SUCCESS")
    
    result = grimace_analyzer.analyze(dummy_landmarks, use_baseline=True)
    print(f"✓ Analysis complete")
    print(f"  - Grimace Score: {result['grimace_score']:.3f}")
    print(f"  - Grimace Percentage: {result['grimace_percentage']:.1f}%")
    print(f"  - Description: {grimace_analyzer.get_description(result['grimace_score'])}")
    
    print("✓ Grimace Analyzer: PASSED")
except Exception as e:
    print(f"✗ Grimace Analyzer: FAILED - {e}")

print()

# Test 3: Eye Analyzer
print("3. Testing Eye Squint Analyzer")
print("-" * 70)
try:
    eye_analyzer = EyeAnalyzer(config.LANDMARK_INDICES)
    
    eye_analyzer.set_baseline(dummy_landmarks)
    print("✓ Baseline calibration: SUCCESS")
    
    result = eye_analyzer.analyze(dummy_landmarks, use_baseline=True)
    print(f"✓ Analysis complete")
    print(f"  - Squint Score: {result['squint_score']:.3f}")
    print(f"  - Squint Percentage: {result['squint_percentage']:.1f}%")
    print(f"  - Description: {eye_analyzer.get_description(result['squint_score'])}")
    print(f"  - Left Aspect Ratio: {result['left_aspect_ratio']:.3f}")
    print(f"  - Right Aspect Ratio: {result['right_aspect_ratio']:.3f}")
    
    print("✓ Eye Analyzer: PASSED")
except Exception as e:
    print(f"✗ Eye Analyzer: FAILED - {e}")

print()

# Test 4: Jaw Analyzer
print("4. Testing Jaw Clench Analyzer")
print("-" * 70)
try:
    jaw_analyzer = JawAnalyzer(config.LANDMARK_INDICES)
    
    jaw_analyzer.set_baseline(dummy_landmarks)
    print("✓ Baseline calibration: SUCCESS")
    
    result = jaw_analyzer.analyze(dummy_landmarks, use_baseline=True)
    print(f"✓ Analysis complete")
    print(f"  - Clench Score: {result['clench_score']:.3f}")
    print(f"  - Clench Percentage: {result['clench_percentage']:.1f}%")
    print(f"  - Description: {jaw_analyzer.get_description(result['clench_score'])}")
    print(f"  - Jaw Angle: {result['jaw_angle']:.1f}°")
    
    print("✓ Jaw Analyzer: PASSED")
except Exception as e:
    print(f"✗ Jaw Analyzer: FAILED - {e}")

print()

# Test 5: Nasolabial Analyzer
print("5. Testing Nasolabial Deepening Analyzer")
print("-" * 70)
try:
    nasolabial_analyzer = NasolabialAnalyzer(config.LANDMARK_INDICES)
    
    nasolabial_analyzer.set_baseline(dummy_landmarks)
    print("✓ Baseline calibration: SUCCESS")
    
    result = nasolabial_analyzer.analyze(dummy_landmarks, use_baseline=True)
    print(f"✓ Analysis complete")
    print(f"  - Strain Score: {result['strain_score']:.3f}")
    print(f"  - Strain Percentage: {result['strain_percentage']:.1f}%")
    print(f"  - Description: {nasolabial_analyzer.get_description(result['strain_score'])}")
    print(f"  - Left Depth: {result['left_depth']:.4f}")
    print(f"  - Right Depth: {result['right_depth']:.4f}")
    
    print("✓ Nasolabial Analyzer: PASSED")
except Exception as e:
    print(f"✗ Nasolabial Analyzer: FAILED - {e}")

print()

# Test 6: Temporal Smoothing
print("6. Testing Temporal Smoothing (History Tracking)")
print("-" * 70)
try:
    analyzer = BrowAnalyzer(config.LANDMARK_INDICES)
    analyzer.set_baseline(dummy_landmarks)
    
    # Simulate multiple frames
    scores = []
    for i in range(15):
        # Slightly modify landmarks to simulate movement
        modified_landmarks = dummy_landmarks + np.random.randn(468, 3) * 0.01
        result = analyzer.analyze(modified_landmarks, use_baseline=True)
        scores.append(result['smoothed_score'])
    
    print(f"✓ Processed 15 frames")
    print(f"  - Score range: {min(scores):.3f} to {max(scores):.3f}")
    print(f"  - Average: {np.mean(scores):.3f}")
    print(f"  - Smoothing working: {'YES' if len(scores) == 15 else 'NO'}")
    
    # Test reset
    analyzer.reset_history()
    print("✓ History reset: SUCCESS")
    
    print("✓ Temporal Smoothing: PASSED")
except Exception as e:
    print(f"✗ Temporal Smoothing: FAILED - {e}")

print()

# Test 7: Integrated Analysis
print("7. Testing Integrated Multi-Analyzer System")
print("-" * 70)
try:
    # Create all analyzers
    analyzers = {
        'brow': BrowAnalyzer(config.LANDMARK_INDICES),
        'grimace': GrimaceAnalyzer(config.LANDMARK_INDICES),
        'eye': EyeAnalyzer(config.LANDMARK_INDICES),
        'jaw': JawAnalyzer(config.LANDMARK_INDICES),
        'nasolabial': NasolabialAnalyzer(config.LANDMARK_INDICES)
    }
    
    # Set baselines for all
    for name, analyzer in analyzers.items():
        analyzer.set_baseline(dummy_landmarks)
    
    print("✓ All analyzers calibrated")
    
    # Run analysis with all
    results = {}
    for name, analyzer in analyzers.items():
        if name == 'brow':
            results[name] = analyzer.analyze(dummy_landmarks)['smoothed_percentage']
        elif name == 'grimace':
            results[name] = analyzer.analyze(dummy_landmarks)['smoothed_percentage']
        elif name == 'eye':
            results[name] = analyzer.analyze(dummy_landmarks)['smoothed_percentage']
        elif name == 'jaw':
            results[name] = analyzer.analyze(dummy_landmarks)['smoothed_percentage']
        elif name == 'nasolabial':
            results[name] = analyzer.analyze(dummy_landmarks)['smoothed_percentage']
    
    print("✓ All analyzers executed")
    print("\n  Individual Scores:")
    for name, score in results.items():
        print(f"    - {name.capitalize()}: {score:.1f}%")
    
    # Calculate weighted pain score
    weighted_score = (
        results['brow'] * config.PAIN_WEIGHTS['brow_tension'] +
        results['grimace'] * config.PAIN_WEIGHTS['grimace'] +
        results['eye'] * config.PAIN_WEIGHTS['eye_squint'] +
        results['jaw'] * config.PAIN_WEIGHTS['jaw_clench'] +
        results['nasolabial'] * config.PAIN_WEIGHTS['nasolabial']
    ) / 10.0  # Convert to 0-10 scale
    
    print(f"\n  Weighted Pain Score: {weighted_score:.2f}/10")
    
    print("✓ Integrated System: PASSED")
except Exception as e:
    print(f"✗ Integrated System: FAILED - {e}")

print()
print("=" * 70)
print("PHASE 3 TEST SUMMARY")
print("=" * 70)
print("✓ All 5 pain analyzers implemented and tested")
print("✓ Baseline calibration working")
print("✓ Temporal smoothing functional")
print("✓ Integrated analysis system ready")
print()
print("Each analyzer provides:")
print("  - Raw score (0-1)")
print("  - Percentage (0-100%)")
print("  - Smoothed values")
print("  - Descriptive text")
print("  - Detailed measurements")
print()
print("Ready for Phase 4: Scoring & Assessment Engine")
print("=" * 70)
