import numpy as np
import os
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockModel:
    """A simple mock model for local testing without MLflow dependencies"""
    
    def __init__(self):
        self.name = "mock_facade_detection_model"
        logger.info(f"Initialized {self.name}")
        
    def predict(self, image):
        """
        Generate mock detection results
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with segmentation data and confidence scores
        """
        # Log image shape if available
        if hasattr(image, 'shape'):
            logger.info(f"Mock model processing image of shape {image.shape}")
            height, width = image.shape[:2]
        else:
            logger.info(f"Mock model processing image without shape")
            height, width = 1024, 1024  # Default size
        
        # Generate a random number of detections (1-5)
        num_detections = random.randint(1, 5)
        segmentations = []
        confidences = []
        
        # Create some mock detections
        for _ in range(num_detections):
            # Random rectangular polygon with 4 points (normalized)
            x1 = random.uniform(0.1, 0.7)
            y1 = random.uniform(0.1, 0.7)
            width_poly = random.uniform(0.1, 0.3)
            height_poly = random.uniform(0.1, 0.3)
            
            segmentation = [
                x1, y1,                         # Top-left
                x1 + width_poly, y1,            # Top-right
                x1 + width_poly, y1 + height_poly,  # Bottom-right
                x1, y1 + height_poly            # Bottom-left
            ]
            
            segmentations.append(segmentation)
            confidences.append(random.uniform(0.75, 0.98))  # High confidence values
        
        return {
            "segmentation": segmentations,
            "confidence": confidences
        }
        
# Create a singleton instance for direct imports
dummy_model = MockModel()
