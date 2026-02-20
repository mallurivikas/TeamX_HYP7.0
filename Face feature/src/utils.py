"""
Utility functions for Pain Detection System
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import json
from pathlib import Path
from datetime import datetime


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: First point [x, y] or [x, y, z]
        point2: Second point [x, y] or [x, y, z]
    
    Returns:
        float: Distance between points
    """
    return np.linalg.norm(point1 - point2)


def calculate_angle(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
    """
    Calculate angle formed by three points (point2 is the vertex)
    
    Args:
        point1: First point
        point2: Vertex point
        point3: Third point
    
    Returns:
        float: Angle in degrees
    """
    vector1 = point1 - point2
    vector2 = point3 - point2
    
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def normalize_landmarks(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Normalize landmarks to 0-1 range based on image dimensions
    
    Args:
        landmarks: Array of landmarks (N, 3) with x, y, z coordinates
        image_shape: (height, width) of the image
    
    Returns:
        np.ndarray: Normalized landmarks
    """
    normalized = landmarks.copy()
    height, width = image_shape
    
    normalized[:, 0] /= width   # Normalize x
    normalized[:, 1] /= height  # Normalize y
    # z is already normalized by MediaPipe
    
    return normalized


def denormalize_landmarks(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert normalized landmarks back to pixel coordinates
    
    Args:
        landmarks: Normalized landmarks (N, 3)
        image_shape: (height, width) of the image
    
    Returns:
        np.ndarray: Denormalized landmarks in pixel coordinates
    """
    denormalized = landmarks.copy()
    height, width = image_shape
    
    denormalized[:, 0] *= width
    denormalized[:, 1] *= height
    
    return denormalized


def calculate_center(points: np.ndarray) -> np.ndarray:
    """
    Calculate the center point of multiple points
    
    Args:
        points: Array of points (N, 2) or (N, 3)
    
    Returns:
        np.ndarray: Center point
    """
    return np.mean(points, axis=0)


def check_lighting_quality(image: np.ndarray) -> Tuple[float, str]:
    """
    Assess image lighting quality
    
    Args:
        image: Input image (BGR or grayscale)
    
    Returns:
        Tuple[float, str]: (brightness_score, quality_description)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate average brightness
    brightness = np.mean(gray)
    
    # Assess quality
    if brightness < 50:
        quality = "Too Dark"
    elif brightness < 100:
        quality = "Poor"
    elif brightness < 150:
        quality = "Fair"
    elif brightness < 200:
        quality = "Good"
    else:
        quality = "Excellent"
    
    return brightness, quality


def save_json(data: dict, filepath: Path):
    """
    Save data to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: Path) -> dict:
    """
    Load data from JSON file
    
    Args:
        filepath: Input file path
    
    Returns:
        dict: Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Create a timestamp string
    
    Args:
        format_str: Datetime format string
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime(format_str)


def resize_image(image: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
    
    Returns:
        np.ndarray: Resized image
    """
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image


def draw_text_with_background(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.6,
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 5
) -> np.ndarray:
    """
    Draw text with background rectangle on image
    
    Args:
        image: Input image
        text: Text to draw
        position: (x, y) position
        font_scale: Font scale
        thickness: Text thickness
        text_color: Text color (BGR)
        bg_color: Background color (BGR)
        padding: Padding around text
    
    Returns:
        np.ndarray: Image with text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(
        image,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )
    
    return image


def get_color_for_score(score: float, thresholds: dict) -> Tuple[int, int, int]:
    """
    Get color based on score and thresholds
    
    Args:
        score: Score value (0-10)
        thresholds: Dictionary with 'low' and 'medium' threshold values
    
    Returns:
        Tuple[int, int, int]: BGR color
    """
    if score < thresholds['low']:
        return (0, 255, 0)  # Green
    elif score < thresholds['medium']:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 0, 255)  # Red


def calculate_percentage(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Convert a value to percentage
    
    Args:
        value: Input value
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        float: Percentage (0-100)
    """
    normalized = (value - min_val) / (max_val - min_val)
    return np.clip(normalized * 100, 0, 100)


def smooth_values(values: List[float], window_size: int = 5) -> List[float]:
    """
    Apply moving average smoothing to values
    
    Args:
        values: List of values
        window_size: Size of smoothing window
    
    Returns:
        List[float]: Smoothed values
    """
    if len(values) < window_size:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        smoothed.append(np.mean(values[start:end]))
    
    return smoothed
