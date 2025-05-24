#!/usr/bin/env python
"""
Create a dummy model for local testing of the Facade AI component.
This script creates a simple MLflow model that returns mock segmentation results.
"""
import os
import sys
import numpy as np
import json
from pathlib import Path

def create_dummy_model(model_path="models/dummy_model"):
    """
    Create a dummy MLflow model structure that can be loaded in local mode
    
    Args:
        model_path: Path where to create the model
    """
    # Ensure model directory exists
    model_dir = Path(model_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "data").mkdir(exist_ok=True)
    
    # Create MLmodel file
    mlmodel_content = """
artifact_path: model
flavors:
  python_function:
    env: conda.yaml
    loader_module: mlflow.pyfunc.model
    python_version: 3.9.0
"""
    with open(model_dir / "MLmodel", "w") as f:
        f.write(mlmodel_content.strip())
        
    # Create conda.yaml
    conda_content = """
name: mlflow-env
channels:
  - defaults
dependencies:
  - python=3.9
  - pip
"""
    with open(model_dir / "conda.yaml", "w") as f:
        f.write(conda_content.strip())
        
    # Create model.py
    model_py_content = """
import numpy as np

class MockModel:
    def predict(self, image):
        # Return mock segmentation results
        return {
            "segmentation": [[0.1, 0.1, 0.8, 0.1, 0.8, 0.8, 0.1, 0.8]],
            "confidence": [0.95]
        }
"""
    with open(model_dir / "data" / "model.py", "w") as f:
        f.write(model_py_content.strip())
        
    # Create a dummy pickle file (not a real pickle, just a placeholder)
    with open(model_dir / "python_model.pkl", "w") as f:
        f.write("dummy_content")
        
    print(f"Created dummy model at {model_path}")
    return model_path

if __name__ == "__main__":
    # Allow specifying model path as argument
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/dummy_model"
    create_dummy_model(model_path)
