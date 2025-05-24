#!/usr/bin/env python
"""
Create a mock model for local testing of the Facade AI component.
"""
import os
import sys
import mlflow
import mlflow.pyfunc
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockSegmentationModel(mlflow.pyfunc.PythonModel):
    """A mock model that returns fake segmentation predictions"""
    
    def predict(self, context, model_input):
        """
        Generate mock segmentation predictions
        
        Args:
            context: MLflow model context
            model_input: Input data (image)
            
        Returns:
            Dictionary with segmentation data
        """
        # Create mock segmentation results (normalized coordinates)
        return {
            "segmentation": [
                [0.1, 0.1, 0.8, 0.1, 0.8, 0.8, 0.1, 0.8]
            ],
            "confidence": [0.95]
        }

def create_mock_model(output_dir="models/mock_model"):
    """
    Create a mock MLflow model for testing
    
    Args:
        output_dir: Directory where to save the model
        
    Returns:
        Path to the created model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Log the model using MLflow
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MockSegmentationModel(),
            conda_env={
                "name": "mock-env",
                "channels": ["defaults"],
                "dependencies": ["python=3.9", "pip", "numpy"]
            }
        )
        run_id = run.info.run_id
        
    # Get the model path
    model_uri = f"runs:/{run_id}/model"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    
    logger.info(f"Created mock model at {local_path}")
    logger.info(f"MLflow run ID: {run_id}")
    logger.info(f"Model URI: {model_uri}")
    
    return local_path, model_uri

if __name__ == "__main__":
    local_path, model_uri = create_mock_model()
    
    # Write the model URI to a file for reference
    with open("mock_model_uri.txt", "w") as f:
        f.write(model_uri)
    
    print(f"Mock model created at: {local_path}")
    print(f"MLflow URI: {model_uri}")
    print("Use the local path or MLflow URI as the model1 parameter")
