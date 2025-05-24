import os
import sys
import logging
import pickle
import traceback
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pkl_model(model_path):
    """
    Load a model from pickle file directly without using MLflow
    
    Args:
        model_path: Path to the model directory containing python_model.pkl
        
    Returns:
        Loaded model object or None if failed
    """
    try:
        # Handle both directory path and direct file path
        if os.path.isdir(model_path):
            pkl_path = os.path.join(model_path, "python_model.pkl")
        else:
            pkl_path = model_path
            
        logger.info(f"Attempting to load model directly from: {pkl_path}")
        
        # Check if file exists
        if not os.path.exists(pkl_path):
            logger.error(f"Model file not found: {pkl_path}")
            return None
            
        # Load the model
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
            
        logger.info(f"Successfully loaded model of type: {type(model)}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None
