#!/bin/bash
# =============================================================================
# setup_dev.sh - Setup script for FacadeAI CPU-based development environment
# =============================================================================
# This script creates a conda environment, directory structure, and dummy model
# required for developing and running the FacadeAI inference component locally
# without requiring GPU resources.
# =============================================================================
set -e

echo "Setting up FacadeAI development environment..."

# Create conda environment from the dev environment file
echo "Creating conda environment from dev_environment.yml..."
conda env create -f dev_environment.yml

# Create required directories if they don't exist
echo "Creating required directories..."
mkdir -p data/images/B3/cam
mkdir -p data/output
mkdir -p models/Dev-Model

# Create a dummy model for testing if one doesn't exist
echo "Checking for dummy model..."
if [ ! -d "models/Dev-Model/data" ]; then
    echo "Creating dummy model for development testing..."
    python -c "
import os
import sys
from pathlib import Path

def create_dummy_model(model_path='models/Dev-Model'):
    # Ensure model directory exists
    model_dir = Path(model_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'data').mkdir(exist_ok=True)
    
    # Create MLmodel file
    mlmodel_content = '''
artifact_path: model
flavors:
  python_function:
    env: conda.yaml
    loader_module: mlflow.pyfunc.model
    python_version: 3.9.0
'''
    with open(model_dir / 'MLmodel', 'w') as f:
        f.write(mlmodel_content.strip())
        
    # Create conda.yaml
    conda_content = '''
name: mlflow-env
channels:
  - defaults
dependencies:
  - python=3.9
  - pip
'''
    with open(model_dir / 'conda.yaml', 'w') as f:
        f.write(conda_content.strip())
        
    # Create model.py
    model_py_content = '''
import numpy as np

class MockModel:
    def predict(self, image):
        # Return mock segmentation results
        return {
            \"segmentation\": [[0.1, 0.1, 0.8, 0.1, 0.8, 0.8, 0.1, 0.8]],
            \"confidence\": [0.95]
        }
'''
    with open(model_dir / 'data' / 'model.py', 'w') as f:
        f.write(model_py_content.strip())
        
    # Create a dummy pickle file (not a real pickle, just a placeholder)
    with open(model_dir / 'python_model.pkl', 'w') as f:
        f.write('dummy_content')
        
    print(f'Created dummy model at {model_path}')
    return model_path

create_dummy_model()
"
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
echo "1. Activate the conda environment: conda activate facadeai_dev"
echo "2. Run the inference script: ./runlocal.sh"
echo "3. Check the output in the data/output directory"
echo ""
echo "For GPU support, install additional dependencies as needed."