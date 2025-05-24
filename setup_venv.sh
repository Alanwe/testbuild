#!/bin/bash
# =============================================================================
# setup_venv.sh - Setup script for FacadeAI using Python venv (no conda)
# =============================================================================
# This script creates a Python virtual environment, installs dependencies, and
# sets up the directory structure and dummy model required for developing and
# running the FacadeAI inference component locally without requiring conda.
# =============================================================================
set -e

echo "Setting up FacadeAI development environment with venv..."

# Create Python virtual environment
echo "Creating Python virtual environment..."
python -m venv .venv

# Activate the environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Create required directories if they don't exist
echo "Creating required directories..."
mkdir -p data/images/B3/cam
mkdir -p data/output
mkdir -p models/Dev-Model

# Create a dummy model for testing if one doesn't exist
echo "Checking for dummy model..."
if [ ! -d "models/Dev-Model/data" ]; then
    echo "Creating dummy model for development testing..."
    python src/old/create_dummy_model.py models/Dev-Model
fi

# Create sample test image if there isn't one
echo "Checking for sample test images..."
if [ ! -f "data/images/B3/cam/sample_image.jpg" ]; then
    echo "Creating sample test image..."
    python -c "
import numpy as np
from PIL import Image
import os

# Create a simple 512x512 image with a rectangle in it
img = np.ones((512, 512, 3), dtype=np.uint8) * 200  # Light gray background
img[100:400, 100:400] = [150, 150, 150]  # Darker rectangle

# Add a simulated defect
img[150:200, 250:350] = [100, 100, 100]  # Even darker rectangle for 'defect'

# Save as JPG
os.makedirs('data/images/B3/cam', exist_ok=True)
Image.fromarray(img).save('data/images/B3/cam/sample_image.jpg')

print('Created sample test image at data/images/B3/cam/sample_image.jpg')
"
fi

echo "Development environment setup complete!"
echo ""
echo "To use this environment:"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Run the inference script: ./dev_runlocal.sh"
echo "3. Check the output in the data/output directory"
echo ""
echo "For GPU support, install additional dependencies as needed."