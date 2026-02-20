"""
Configuration settings for Pain Detection System
Adjust these parameters to fine-tune the system behavior
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
TEST_DIR = BASE_DIR / "tests"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, TEST_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MEDIAPIPE SETTINGS
# ============================================================================
MEDIAPIPE_CONFIG = {
    "max_num_faces": 1,
    "refine_landmarks": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# ============================================================================
# FACIAL LANDMARKS FOR PAIN INDICATORS
# ============================================================================
LANDMARK_INDICES = {
    "eyebrow_left": [70, 107],
    "eyebrow_right": [336, 300],
    "mouth_corners": [61, 291],
    "mouth_center": [0, 17],
    "eye_left": [159, 145],
    "eye_right": [386, 374],
    "jaw_left": [234, 93],
    "jaw_right": [454, 323],
    "nasolabial_left": [48],
    "nasolabial_right": [278],
}

# ============================================================================
# PAIN SCORING WEIGHTS
# ============================================================================
PAIN_WEIGHTS = {
    "brow_tension": 0.25,      # 25% contribution to overall score
    "grimace": 0.30,            # 30% contribution
    "eye_squint": 0.20,         # 20% contribution
    "jaw_clench": 0.20,         # 20% contribution
    "nasolabial": 0.05,         # 5% contribution
}

# ============================================================================
# THRESHOLDS
# ============================================================================
PAIN_THRESHOLDS = {
    "low": 3.0,      # Pain score < 3: Green (low pain)
    "medium": 6.0,   # Pain score 3-6: Yellow (moderate pain)
    "high": 10.0,    # Pain score > 6: Red (high pain)
}

CONFIDENCE_THRESHOLDS = {
    "high": 0.80,    # >= 80% confidence
    "medium": 0.60,  # 60-80% confidence
    "low": 0.40,     # < 60% confidence
}

# Quality thresholds
FACE_DETECTION_QUALITY = {
    "excellent": 0.9,
    "good": 0.75,
    "fair": 0.6,
    "poor": 0.4,
}

LIGHTING_QUALITY_THRESHOLD = 50  # Minimum average brightness

# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================
TEMPORAL_WINDOW_SECONDS = 30  # Track pain over last 30 seconds
UPDATE_FREQUENCY = 0.5         # Update every 0.5 seconds
BUFFER_SIZE = int(TEMPORAL_WINDOW_SECONDS / UPDATE_FREQUENCY)  # 60 samples

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================
DISPLAY_CONFIG = {
    "window_name": "Pain Detection System",
    "window_width": 1280,
    "window_height": 720,
    "fps": 30,
    "show_landmarks": True,
    "show_indicators": True,
    "show_score": True,
}

# Color codes (BGR format for OpenCV)
COLORS = {
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
}

# ============================================================================
# REPORT SETTINGS
# ============================================================================
REPORT_CONFIG = {
    "save_format": ["json", "txt", "image"],
    "include_landmarks": True,
    "include_temporal_data": True,
    "timestamp_format": "%Y-%m-%d %H:%M:%S",
}

# ============================================================================
# BASELINE CALIBRATION
# ============================================================================
BASELINE_CONFIG = {
    "required": False,           # Whether baseline is mandatory
    "capture_duration": 3.0,     # Seconds to capture neutral expression
    "num_samples": 30,           # Number of frames to average
}

# ============================================================================
# ERROR HANDLING
# ============================================================================
ERROR_MESSAGES = {
    "no_face": "No face detected. Please ensure your face is visible and well-lit.",
    "multiple_faces": "Multiple faces detected. Please ensure only one person is in frame.",
    "poor_lighting": "Poor lighting conditions detected. Please improve lighting.",
    "partial_face": "Face partially obscured. Please ensure full face visibility.",
    "low_confidence": "Low detection confidence. Results may be unreliable.",
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": OUTPUT_DIR / "pain_detection.log",
}

# ============================================================================
# VALIDATION & TESTING
# ============================================================================
VALIDATION_CONFIG = {
    "test_mode": False,
    "benchmark_mode": False,
    "save_debug_images": False,
}
