"""
Setup Verification Script
Run this to verify all dependencies are installed correctly
"""

import sys
from pathlib import Path

print("=" * 70)
print("PAIN DETECTION SYSTEM - SETUP VERIFICATION")
print("=" * 70)
print()

# Check Python version
print(f"✓ Python Version: {sys.version}")
print()

# Check dependencies
dependencies = {
    "mediapipe": "MediaPipe Face Mesh",
    "cv2": "OpenCV",
    "numpy": "NumPy",
    "matplotlib": "Matplotlib",
    "PIL": "Pillow",
    "dateutil": "Python-dateutil"
}

print("Checking Dependencies:")
print("-" * 70)

all_installed = True
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"✓ {name:25} - Installed")
    except ImportError:
        print(f"✗ {name:25} - NOT FOUND")
        all_installed = False

print()

# Check directory structure
print("Checking Directory Structure:")
print("-" * 70)

base_dir = Path(__file__).parent
required_dirs = {
    "src": "Source code directory",
    "src/pain_analyzers": "Pain analyzer modules",
    "tests": "Testing directory",
    "tests/test_images": "Test images directory",
    "outputs": "Output reports directory",
    "data": "Baseline data directory"
}

all_dirs_exist = True
for dir_path, description in required_dirs.items():
    full_path = base_dir / dir_path
    if full_path.exists():
        print(f"✓ {description:30} - OK")
    else:
        print(f"✗ {description:30} - MISSING")
        all_dirs_exist = False

print()

# Check configuration files
print("Checking Configuration Files:")
print("-" * 70)

config_files = {
    "config.py": "Main configuration",
    "requirements.txt": "Dependencies list",
    "README.md": "Documentation",
    ".gitignore": "Git ignore rules"
}

all_files_exist = True
for file_path, description in config_files.items():
    full_path = base_dir / file_path
    if full_path.exists():
        print(f"✓ {description:30} - OK")
    else:
        print(f"✗ {description:30} - MISSING")
        all_files_exist = False

print()
print("=" * 70)

if all_installed and all_dirs_exist and all_files_exist:
    print("✓ ALL CHECKS PASSED - System is ready!")
    print()
    print("Next Steps:")
    print("1. Place test images in tests/test_images/")
    print("2. Run: python src/main.py --help")
    print("3. Start with: python src/main.py --mode webcam")
else:
    print("✗ SETUP INCOMPLETE - Please fix the issues above")
    print()
    if not all_installed:
        print("Run: pip install -r requirements.txt")

print("=" * 70)
