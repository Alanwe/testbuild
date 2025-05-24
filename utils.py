"""
Utility functions for Facade AI Inference Component
"""
import os
import json
import time
import logging
import warnings
import inspect
import functools
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import traceback
import mlflow
import cv2
from PIL import Image
import yaml
from mlflow.types import DataType
from mlflow.types.schema import Schema
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Configure logging
logger = logging.getLogger(__name__)

# Suppress specific warnings that don't affect functionality
warnings.filterwarnings("ignore", message="Detected one or more mismatches between the model's dependencies")
warnings.filterwarnings("ignore", message="Model's `predict` method contains invalid parameters")
warnings.filterwarnings("ignore", message="Failure while loading azureml_run_type_providers")

# Category ID mapping for different models
MODEL_CATEGORY_IDS = {
    "stonefractions": 1,
    "windows_detection": 2,
    "facade_defects": 3,
    "glazing-defects": 3,
    # Add more model name to category ID mappings as needed
}

def check_model_dependencies(model_path):
    """
    Check model dependencies and log any mismatches
    
    Args:
        model_path: Path to the MLflow model
    """
    try:
        # Get model dependencies
        deps = mlflow.pyfunc.get_model_dependencies(model_path)
        if deps:
            logger.info(f"Model dependencies: {deps}")
            
            # Check for mismatches with the current environment
            import pkg_resources
            installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
            
            mismatches = []
            if hasattr(deps, 'conda_env') and 'dependencies' in deps.conda_env:
                for dep in deps.conda_env['dependencies']:
                    if isinstance(dep, dict) and 'pip' in dep:
                        for pip_dep in dep['pip']:
                            # Simple parsing of pip requirements
                            parts = pip_dep.replace('==', '=').replace('>=', '=').replace('<=', '=').split('=')
                            if len(parts) >= 2:
                                pkg_name = parts[0].lower()
                                required_version = parts[1]
                                
                                if pkg_name in installed_packages:
                                    installed_version = installed_packages[pkg_name]
                                    if installed_version != required_version:
                                        mismatches.append(f"{pkg_name}: required={required_version}, installed={installed_version}")
            
            if mismatches:
                logger.warning(f"Dependency mismatches detected: {mismatches}")
            else:
                logger.info("No dependency mismatches detected")
                
    except Exception as e:
        logger.warning(f"Could not check model dependencies: {str(e)}")

def sliding_window(image, window_size: int = 512, overlap: int = 64) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Split an image into overlapping windows of specified size
    
    Args:
        image: Input image as numpy array
        window_size: Size of each window
        overlap: Overlap between adjacent windows
    
    Returns:
        List of (window, (x, y)) tuples where (x, y) is the top-left corner
    """
    height, width = image.shape[:2]
    windows = []
    
    stride = window_size - overlap
    
    for y in range(0, height - overlap, stride):
        for x in range(0, width - overlap, stride):
            # Make sure windows don't go beyond image bounds
            end_x = min(x + window_size, width)
            end_y = min(y + window_size, height)
            
            # Handle edge cases - make windows the full size if possible
            if end_x - x < window_size and x > 0:
                x = max(0, end_x - window_size)
            if end_y - y < window_size and y > 0:
                y = max(0, end_y - window_size)
                
            # Extract the window
            window = image[y:end_y, x:end_x]
            
            # Skip if window is too small (edge case)
            if window.shape[0] < window_size/2 or window.shape[1] < window_size/2:
                continue
                
            # Resize if window is not the expected size
            if window.shape[0] != window_size or window.shape[1] != window_size:
                window = cv2.resize(window, (window_size, window_size))
                
            windows.append((window, (x, y)))
            
    logger.info(f"Generated {len(windows)} sliding windows")
    return windows

def load_image_as_array(image_path: str) -> np.ndarray:
    """
    Load an image from disk as a numpy array using PIL for better compatibility
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array in BGR format (OpenCV format)
    """
    try:
        # Open the image with PIL
        with Image.open(image_path) as img:
            # Convert to RGB and then to numpy array
            img_array = np.array(img.convert('RGB'))
            
            # Convert from RGB to BGR (OpenCV format)
            bgr_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return bgr_image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None

def load_image_as_bytes(image_path: str) -> bytes:
    """
    Load an image from disk as bytes
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as bytes
    """
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading image as bytes {image_path}: {str(e)}")
        return None

def get_input_schema(model_path: str):
    """
    Get the input schema for a model
    
    Args:
        model_path: Path to the model
        
    Returns:
        Model schema if available, None otherwise
    """
    try:
        if not model_path or not os.path.exists(str(model_path)):
            logger.error(f"Model path does not exist: {model_path}")
            return None
            
        # Use MLflow to get model signature
        try:
            model_info = mlflow.pyfunc.load_model(model_path, suppress_warnings=True)
            if hasattr(model_info, 'metadata') and hasattr(model_info.metadata, 'signature'):
                return model_info.metadata.signature.inputs
            else:
                # No schema information available
                logger.warning(f"No schema information available for model: {model_path}")
                return None
        except Exception as e:
            logger.error(f"Error getting model schema: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error getting model schema: {str(e)}")
        return None

def log_model_details(ml_model):
    """Log MLmodel file details."""
    if "flavors" in ml_model:
        logger.info(f"Model flavors: {list(ml_model['flavors'].keys())}")
    
    if "metadata" in ml_model:
        logger.info(f"Model metadata available")

def get_dtypes(schema):
    """
    Extract data types from schema
    
    Args:
        schema: Model schema
        
    Returns:
        Tuple of (data_type, shape) or None if schema is not available
    """
    if schema is None:
        return None
    
    try:
        # Handle different types of schemas
        if hasattr(schema, 'is_tensor_spec') and schema.is_tensor_spec():
            # Get expected shape and dtype
            sample_input = schema.sample()
            if hasattr(sample_input, 'dtype') and hasattr(sample_input, 'shape'):
                return (sample_input.dtype, sample_input.shape)
        elif hasattr(schema, 'sample'):
            # Try to get sample input even without is_tensor_spec
            sample_input = schema.sample()
            if hasattr(sample_input, 'dtype') and hasattr(sample_input, 'shape'):
                return (sample_input.dtype, sample_input.shape)
        
        # For non-tensor schemas, just return None
        return None
    except Exception as e:
        logger.error(f"Error extracting dtype from schema: {str(e)}")
        return None

def nparray_tolist(array):
    """Convert numpy array to JSON-serializable list"""
    if isinstance(array, np.ndarray):
        if array.dtype == object:  # Handle nested arrays
            return [_get_jsonable_obj(item) for item in array]
        return _get_jsonable_obj(array)
    return array

def create_model_adapter(model):
    """
    Create a function adapter for model prediction that handles parameter mismatches.
    This fixes the 'input_data' vs 'model_input' parameter issue.
    """
    # Try to get the signature of the model's predict method
    original_predict = model.predict
    sig = inspect.signature(original_predict)
    params = list(sig.parameters.keys())
    
    # Log the predict method signature for debugging
    logger.info(f"Original predict method signature: {params}")
    
    # Create a wrapped predict function that maps parameters correctly
    @functools.wraps(original_predict)
    def wrapped_predict(data, **kwargs):
        try:
            # First try direct call with data
            return original_predict(data)
        except Exception as e1:
            logger.info(f"Direct predict failed: {str(e1)}, trying alternative methods")
            
            try:
                # Try with model_input parameter (MLflow standard)
                if 'model_input' in params:
                    return original_predict(model_input=data)
                # Try with input_data parameter (common in custom models)
                elif 'input_data' in params:
                    return original_predict(input_data=data)
                # Try with input parameter
                elif 'input' in params:
                    return original_predict(input=data)
                # Try with data parameter
                elif 'data' in params:
                    return original_predict(data=data)
                # Try with X parameter (scikit-learn style)
                elif 'X' in params:
                    return original_predict(X=data)
                # Try with dictionary unpacking if all else fails
                else:
                    if isinstance(data, dict):
                        return original_predict(**data)
                    else:
                        # Try common parameter names as kwargs
                        for param_name in ['data', 'model_input', 'input_data', 'input', 'X']:
                            try:
                                kwargs[param_name] = data
                                return original_predict(**kwargs)
                            except:
                                continue
                        
                        # Last resort - try positional argument
                        return original_predict(data)
            except Exception as e2:
                logger.error(f"All prediction parameter mappings failed: {str(e2)}")
                raise
    
    return wrapped_predict

def prepare_image_input(image_data, schema=None):
    """
    Prepare image data according to the model's schema requirements
    
    Args:
        image_data: Input image data (numpy array)
        schema: Model input schema
        
    Returns:
        Properly formatted input data for the model
    """
    try:
        if schema is None:
            # If no schema, just return the image data as is
            return image_data
            
        # Check for binary schema
        if len(schema.inputs) == 1:
            input_type = schema.input_types()[0]
            
            # For binary input type, convert to bytes
            if input_type == DataType.binary:
                logger.info("Converting image to binary format for binary schema")
                is_success, buffer = cv2.imencode(".jpg", image_data)
                if not is_success:
                    raise ValueError("Failed to encode image")
                    
                # For binary column spec, return a dataframe with the binary data
                import pandas as pd
                return pd.DataFrame({schema.input_names()[0]: [buffer.tobytes()]})
                
            # For tensor input, check if reshaping is needed
            elif schema.is_tensor_spec():
                data_type, data_shape = schema.numpy_types()[0], schema.inputs[0].shape
                logger.info(f"Preparing tensor input with type {data_type} and shape {data_shape}")
                
                # If shape is (-1,), create a batch of 1 with object dtype
                if data_shape == (-1,):
                    batch = np.empty(1, dtype=data_type)
                    batch[0] = image_data
                    return batch
                    
                # If shape requires reshaping, try to reshape
                if len(data_shape) != len(image_data.shape):
                    # Create batch dimension if needed
                    if len(data_shape) == 4 and len(image_data.shape) == 3:
                        return np.expand_dims(image_data.astype(data_type), axis=0)
                    elif len(data_shape) == 2 and len(image_data.shape) == 3:
                        # Model expects 2D input but we have 3D image
                        # Convert to grayscale
                        logger.info("Converting 3D color image to 2D grayscale")
                        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                        return gray.astype(data_type)
                
                # Otherwise, just convert type
                return image_data.astype(data_type)
                
        # Default: return original image
        return image_data
            
    except Exception as e:
        logger.error(f"Error preparing image input: {str(e)}")
        return image_data

def run_generic_model_predict(model, image_data, confidence_threshold=50):
    """
    Run model prediction in the simplest possible way to avoid syntax errors
    
    Args:
        model: MLflow model object
        image_data: Input image data (numpy array)
        confidence_threshold: Confidence threshold for filtering predictions
        
    Returns:
        List of detection dictionaries with standardized format
    """
    try:
        # Just call predict directly - simplest approach from good.py
        predict_result = model.predict(image_data)
        
        # Log the result type for debugging
        logger.info(f"Prediction result type: {type(predict_result)}")
        
        # Initialize empty detections list
        detections = []
        
        # Handle dictionary results (most common for instance segmentation)
        if isinstance(predict_result, dict):
            logger.info(f"Dict result keys: {list(predict_result.keys())}")
            
            # Handle masks and scores format
            if 'masks' in predict_result and 'scores' in predict_result:
                masks = predict_result.get('masks', [])
                scores = predict_result.get('scores', [])
                classes = predict_result.get('classes', [1] * len(scores))
                
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    if score * 100 < confidence_threshold:
                        continue
                        
                    class_id = int(classes[i]) if i < len(classes) else 1
                    
                    if isinstance(mask, np.ndarray):
                        try:
                            # Convert mask to binary
                            binary_mask = (mask > 0.5).astype(np.uint8) if mask.dtype in [np.float32, np.float64] else mask.astype(np.uint8)
                            
                            # Find contours
                            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                segmentation = largest_contour.flatten().tolist()
                                
                                detection = {
                                    "segmentation": segmentation,
                                    "score": float(score),
                                    "class_id": class_id
                                }
                                detections.append(detection)
                        except Exception as e:
                            logger.error(f"Error processing mask: {str(e)}")
            
            # Handle direct segmentation format
            elif 'segmentation' in predict_result:
                segmentations = predict_result.get('segmentation', [])
                scores = predict_result.get('scores', [1.0] * len(segmentations))
                classes = predict_result.get('classes', [1] * len(segmentations))
                
                for i, segmentation in enumerate(segmentations):
                    score = scores[i] if i < len(scores) else 1.0
                    
                    if score * 100 < confidence_threshold:
                        continue
                        
                    class_id = classes[i] if i < len(classes) else 1
                    
                    detection = {
                        "segmentation": segmentation,
                        "score": float(score),
                        "class_id": class_id
                    }
                    detections.append(detection)
            
            # Handle predictions key format
            elif 'predictions' in predict_result:
                predictions = predict_result['predictions']
                
                if isinstance(predictions, list):
                    for pred in predictions:
                        if isinstance(pred, dict) and 'segmentation' in pred:
                            score = pred.get('score', 1.0)
                            
                            if score * 100 < confidence_threshold:
                                continue
                                
                            class_id = pred.get('class_id', 1)
                            
                            detection = {
                                "segmentation": pred['segmentation'],
                                "score": float(score),
                                "class_id": class_id
                            }
                            detections.append(detection)
        
        # Handle list results
        elif isinstance(predict_result, list):
            logger.info(f"List result with {len(predict_result)} items")
            
            for item in predict_result:
                if isinstance(item, dict):
                    # Check for segmentation key
                    if 'segmentation' in item:
                        score = item.get('score', 1.0)
                        
                        if score * 100 < confidence_threshold:
                            continue
                            
                        class_id = item.get('class_id', 1)
                        
                        detection = {
                            "segmentation": item['segmentation'],
                            "score": float(score),
                            "class_id": class_id
                        }
                        detections.append(detection)
                    # Check for mask key
                    elif 'mask' in item or 'masks' in item:
                        mask_key = 'mask' if 'mask' in item else 'masks'
                        mask = item[mask_key]
                        score = item.get('score', 1.0)
                        
                        if score * 100 < confidence_threshold:
                            continue
                            
                        # Process the mask
                        try:
                            if isinstance(mask, np.ndarray):
                                binary_mask = (mask > 0.5).astype(np.uint8) if mask.dtype in [np.float32, np.float64] else mask.astype(np.uint8)
                                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                if contours:
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    segmentation = largest_contour.flatten().tolist()
                                    
                                    detection = {
                                        "segmentation": segmentation,
                                        "score": float(score),
                                        "class_id": item.get('class_id', 1)
                                    }
                                    detections.append(detection)
                        except Exception as e:
                            logger.error(f"Error processing mask: {str(e)}")
        
        # Handle numpy array results
        elif isinstance(predict_result, np.ndarray):
            logger.info(f"Numpy array result with shape {predict_result.shape}")
            
            # Handle masks format
            if len(predict_result.shape) == 3 and predict_result.shape[0] > 0:
                for i in range(predict_result.shape[0]):
                    mask = predict_result[i]
                    
                    try:
                        # Convert mask to binary
                        binary_mask = (mask > 0.5).astype(np.uint8) if mask.dtype in [np.float32, np.float64] else mask.astype(np.uint8)
                        
                        # Find contours
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            segmentation = largest_contour.flatten().tolist()
                            
                            detection = {
                                "segmentation": segmentation,
                                "score": 1.0,  # Default score
                                "class_id": 1   # Default class
                            }
                            detections.append(detection)
                    except Exception as e:
                        logger.error(f"Error processing mask array: {str(e)}")
        
        logger.info(f"Extracted {len(detections)} detections")
        return detections
        
    except Exception as e:
        logger.error(f"Error in run_generic_model_predict: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        return []

def filter_by_confidence(predictions, confidence_threshold):
    """
    Filter predictions by confidence threshold
    
    Args:
        predictions: List of predictions
        confidence_threshold: Confidence threshold (0-100)
        
    Returns:
        Filtered predictions
    """
    # Convert threshold to decimal if needed
    thresh = confidence_threshold / 100.0 if confidence_threshold > 1 else confidence_threshold
    
    try:
        # Handle different prediction formats
        if isinstance(predictions, list):
            filtered = []
            for pred in predictions:
                if isinstance(pred, dict):
                    # Check for common confidence field names
                    confidence = None
                    for field in ['confidence', 'score', 'probability', 'conf']:
                        if field in pred:
                            confidence = pred[field]
                            break
                    
                    # If confidence is found and above threshold, keep the prediction
                    if confidence is not None:
                        # Handle percentage vs decimal representations
                        if confidence > 1 and confidence <= 100:
                            confidence = confidence / 100.0
                        
                        if confidence >= thresh:
                            filtered.append(pred)
                else:
                    # If not a dict, just include it
                    filtered.append(pred)
            return filtered
        
        # Default: return the original predictions
        return predictions
    
    except Exception as e:
        logger.error(f"Error filtering predictions: {str(e)}")
        return predictions

def create_coordinate_pairs(segmentation):
    """Convert flat segmentation list to list of coordinate pairs"""
    if not segmentation or len(segmentation) < 2:
        return []
        
    pairs = []
    for i in range(0, len(segmentation), 2):
        if i+1 < len(segmentation):
            pairs.append((segmentation[i], segmentation[i+1]))
    return pairs

def flatten_coordinates(coords):
    """Convert list of coordinate pairs to flat list"""
    if not coords:
        return []
        
    flat = []
    for x, y in coords:
        flat.extend([x, y])
    return flat

def recalculate_coordinates(segmentation, offset):
    """
    Recalculate segmentation coordinates based on sliding window offset
    
    Args:
        segmentation: Original segmentation coordinates
        offset: (x, y) offset of the sliding window
        
    Returns:
        Recalculated segmentation coordinates
    """
    offset_x, offset_y = offset
    
    # Handle different segmentation formats
    if isinstance(segmentation, list):
        # Handle nested lists (multiple polygons)
        if len(segmentation) > 0 and isinstance(segmentation[0], list):
            return [recalculate_coordinates(poly, offset) for poly in segmentation]
        
        # Single polygon as a flat list of coordinates [x1, y1, x2, y2, ...]
        recalculated = []
        for i in range(0, len(segmentation), 2):
            if i+1 < len(segmentation):
                recalculated.append(segmentation[i] + offset_x)
                recalculated.append(segmentation[i+1] + offset_y)
        return recalculated
    
    # Handle numpy arrays
    elif isinstance(segmentation, np.ndarray):
        if len(segmentation.shape) == 2 and segmentation.shape[1] == 2:
            # Array of [x, y] points
            return segmentation + np.array([offset_x, offset_y])
        else:
            # Try to convert to flat list and process
            return recalculate_coordinates(segmentation.flatten().tolist(), offset)
    
    logger.warning(f"Unsupported segmentation format: {type(segmentation)}")
    return segmentation

def flatten_polygon(polygon):
    """
    Flatten nested polygon structures to a single flat list
    
    Args:
        polygon: Polygon coordinates, possibly nested
        
    Returns:
        Flattened list of coordinates [x1, y1, x2, y2, ...]
    """
    if not polygon:
        return []
        
    # Already flat list with even number of elements
    if isinstance(polygon, list) and len(polygon) % 2 == 0 and all(isinstance(x, (int, float)) for x in polygon):
        return polygon
    
    # Handle numpy arrays
    if isinstance(polygon, np.ndarray):
        if len(polygon.shape) == 2 and polygon.shape[1] == 2:
            # Convert 2D array of points to flat list
            return polygon.flatten().tolist()
        else:
            # Try to convert to flat list
            return flatten_polygon(polygon.tolist())
    
    # Handle nested lists
    if isinstance(polygon, list):
        # Nested structure with one level [[x1, y1], [x2, y2], ...]
        if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in polygon):
            return [coord for point in polygon for coord in point]
        
        # Multiple polygons [[poly1], [poly2], ...]
        if len(polygon) > 0 and isinstance(polygon[0], list):
            # Take the first polygon if there are multiple
            return flatten_polygon(polygon[0])
    
    logger.warning(f"Couldn't flatten polygon: {polygon}")
    return polygon if isinstance(polygon, list) else []

def calculate_bbox_from_segmentation(segmentation):
    """
    Calculate bounding box from segmentation coordinates
    
    Args:
        segmentation: Flat list of coordinates [x1, y1, x2, y2, ...]
        
    Returns:
        Bounding box [x, y, width, height]
    """
    if not segmentation or len(segmentation) < 4:
        return [0, 0, 0, 0]
    
    # Extract x and y coordinates
    x_coords = segmentation[0::2]
    y_coords = segmentation[1::2]
    
    # Calculate bbox
    x_min = min(x_coords)
    y_min = min(y_coords)
    width = max(x_coords) - x_min
    height = max(y_coords) - y_min
    
    return [x_min, y_min, width, height]

def calculate_polygon_area(segmentation):
    """
    Calculate area of a polygon
    
    Args:
        segmentation: Flat list of coordinates [x1, y1, x2, y2, ...]
        
    Returns:
        Area of the polygon
    """
    if not segmentation or len(segmentation) < 6:  # Need at least 3 points
        return 0
    
    # Convert to list of points [(x1, y1), (x2, y2), ...]
    points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
    
    # Calculate area using Shoelace formula
    area = 0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    
    return abs(area) / 2

def multiply_coordinates(segmentation, scale_x, scale_y):
    """
    Multiply coordinates by scale factors
    
    Args:
        segmentation: Flat list of coordinates [x1, y1, x2, y2, ...]
        scale_x: Scale factor for x coordinates
        scale_y: Scale factor for y coordinates
        
    Returns:
        Scaled coordinates
    """
    if not segmentation or len(segmentation) < 2:
        return segmentation
    
    scaled = []
    for i in range(0, len(segmentation), 2):
        if i+1 < len(segmentation):
            scaled.append(segmentation[i] * scale_x)
            scaled.append(segmentation[i+1] * scale_y)
    
    return scaled

def get_cosmosdb_client(connection_string=None, key_vault_url=None):
    """Get CosmosDB client using either connection string or Key Vault"""
    if connection_string:
        return CosmosClient.from_connection_string(connection_string)
    
    if key_vault_url:
        try:
            # Get credentials from Azure
            credential = DefaultAzureCredential()
            secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
            
            # Get the CosmosDB connection string from Key Vault
            cosmos_connection_string = secret_client.get_secret("CosmosDBConnectionString").value
            return CosmosClient.from_connection_string(cosmos_connection_string)
        except Exception as e:
            logger.error(f"Failed to get CosmosDB connection string from Key Vault: {str(e)}")
            return None
    
    return None

def get_processed_ids(cosmos_client, batch_id, database_name="FacadeDB", container_name="Images"):
    """Get list of already processed image IDs from CosmosDB"""
    try:
        database = cosmos_client.get_database_client(database_name)
        container = database.get_container_client(container_name)
        
        # Query for images in this batch that have already been processed
        query = f"SELECT c.id FROM c WHERE c.BatchID = '{batch_id}' AND c.Status != ''"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        # Extract the IDs
        processed_ids = [item['id'] for item in items]
        return processed_ids
    except Exception as e:
        logger.error(f"Error getting processed IDs: {str(e)}")
        return []

def update_cosmos_db(cosmos_client, image_id, annotations, batch_id=None, 
                    database_name="FacadeDB", container_name="Images"):
    """Update a CosmosDB record with new annotations"""
    try:
        database = cosmos_client.get_database_client(database_name)
        container = database.get_container_client(container_name)
        
        # Try to get the existing record
        try:
            item = container.read_item(item=image_id, partition_key=image_id)
            logger.info(f"Found existing record for {image_id}")
        except Exception as e:
            logger.warning(f"No existing record found for {image_id}, creating new record")
            # Create a basic record if it doesn't exist
            item = {
                "id": image_id,
                "ImageID": image_id,
                "BatchID": batch_id,
                "Status": "",
                "info": {
                    "description": "Facade Studio Annotations",
                    "date_created": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())
                },
                "images": [
                    {
                        "id": image_id,
                        "file_name": f"{batch_id}/cam/{image_id}.jpg",
                        "width": 3840,  # Default values, could be updated
                        "height": 2160
                    }
                ],
                "annotations": [],
                "categories": [
                    {"id": 1, "name": "Stone-Fracture", "supercategory": "Facade"},
                    {"id": 2, "name": "Window-Defect", "supercategory": "Facade"},
                    {"id": 3, "name": "Facade-Defect", "supercategory": "Facade"}
                ]
            }
        
        # Update annotations
        if "annotations" not in item:
            item["annotations"] = []
        
        # Add new annotations
        item["annotations"].extend(annotations)
        
        # Update status
        item["Status"] = "Processed"
        
        # Update the record
        container.upsert_item(item)
        logger.info(f"Updated CosmosDB record for {image_id} with {len(annotations)} annotations")
        return True
    except Exception as e:
        logger.error(f"Error updating CosmosDB: {str(e)}")
        return False

def save_local_results(output_dir, image_id, annotations):
    """Save results locally for debugging or local mode"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{image_id}_annotations.json")
        
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        logger.info(f"Saved annotations to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving local results: {str(e)}")
        return False

def log_inference_metrics(model_name, inference_time, num_detections, local_mode=False):
    """Log model inference metrics to MLflow"""
    try:
        # Skip MLflow logging in local mode
        if local_mode:
            logger.info(f"Local mode: Skipping MLflow logging for {model_name}: inference_time={inference_time:.2f}s, detections={num_detections}")
            return
            
        mlflow.log_metric(f"{model_name}_inference_time", inference_time)
        mlflow.log_metric(f"{model_name}_num_detections", num_detections)
        logger.info(f"Logged metrics for {model_name}: inference_time={inference_time:.2f}s, detections={num_detections}")
    except Exception as e:
        logger.warning(f"Failed to log metrics: {str(e)}")

# Schema and model input handling functions adapted from good.py
def get_input_schema(model_path):
    """Get input schema from MLmodel file"""
    try:
        ml_model_path = os.path.join(model_path, "MLmodel")
        if not os.path.exists(ml_model_path):
            logger.warning(f"MLmodel file not found at {ml_model_path}")
            return None
            
        with open(ml_model_path, "r") as stream:
            ml_model = yaml.safe_load(stream)
        
        # Log model details if available (fix for line 213)
        if "flavors" in ml_model:
            logger.info(f"Model flavors: {list(ml_model['flavors'].keys())}")
        
        if "metadata" in ml_model:
            logger.info(f"Model metadata available")
        
        if "signature" in ml_model:
            if "inputs" in ml_model["signature"]:
                return Schema.from_json(ml_model["signature"]["inputs"])
        
        logger.warning("No signature present in MLmodel file")
        return None
    except Exception as e:
        logger.error(f'Error reading model signature: {str(e)}')
        return None

def get_dtypes(schema):
    """Get data types from schema"""
    try:
        if schema is None:
            return None
        elif schema.is_tensor_spec():
            data_type = schema.numpy_types()[0]
            data_shape = schema.inputs[0].shape
            logger.info(f'Model expects tensor type: {data_type} and shape: {data_shape}')
            return data_type, data_shape
        else:                
            column_dtypes = dict(zip(schema.input_names(), schema.pandas_types()))
            logger.info(f'Model expects datatypes: {column_dtypes}')
            return column_dtypes
    except Exception as e:
        logger.error(f'Error reading types from schema: {str(e)}')
        return None

def nparray_tolist(array):
    """Convert numpy array to JSON-serializable list"""
    if isinstance(array, np.ndarray):
        if array.dtype == object:  # Handle nested arrays
            return [_get_jsonable_obj(item) for item in array]
        return _get_jsonable_obj(array)
    return array

def create_model_adapter(model):
    """
    Create a function adapter for model prediction that handles parameter mismatches.
    This fixes the 'input_data' vs 'model_input' parameter issue.
    """
    # Try to get the signature of the model's predict method
    original_predict = model.predict
    sig = inspect.signature(original_predict)
    params = list(sig.parameters.keys())
    
    # Log the predict method signature for debugging
    logger.info(f"Original predict method signature: {params}")
    
    # Create a wrapped predict function that maps parameters correctly
    @functools.wraps(original_predict)
    def wrapped_predict(data, **kwargs):
        try:
            # First try direct call with data
            return original_predict(data)
        except Exception as e1:
            logger.info(f"Direct predict failed: {str(e1)}, trying alternative methods")
            
            try:
                # Try with model_input parameter (MLflow standard)
                if 'model_input' in params:
                    return original_predict(model_input=data)
                # Try with input_data parameter (common in custom models)
                elif 'input_data' in params:
                    return original_predict(input_data=data)
                # Try with input parameter
                elif 'input' in params:
                    return original_predict(input=data)
                # Try with data parameter
                elif 'data' in params:
                    return original_predict(data=data)
                # Try with X parameter (scikit-learn style)
                elif 'X' in params:
                    return original_predict(X=data)
                # Try with dictionary unpacking if all else fails
                else:
                    if isinstance(data, dict):
                        return original_predict(**data)
                    else:
                        # Try common parameter names as kwargs
                        for param_name in ['data', 'model_input', 'input_data', 'input', 'X']:
                            try:
                                kwargs[param_name] = data
                                return original_predict(**kwargs)
                            except Exception:
                                continue
                        
                        # Last resort - try positional argument
                        return original_predict(data)
            except Exception as e2:
                logger.error(f"All prediction parameter mappings failed: {str(e2)}")
                raise
    
    return wrapped_predict

def prepare_image_input(image_data, schema=None):
    """
    Prepare image data according to the model's schema requirements
    
    Args:
        image_data: Input image data (numpy array)
        schema: Model input schema
        
    Returns:
        Properly formatted input data for the model
    """
    try:
        if schema is None:
            # If no schema, just return the image data as is
            return image_data
            
        # Check for binary schema
        if len(schema.inputs) == 1:
            input_type = schema.input_types()[0]
            
            # For binary input type, convert to bytes
            if input_type == DataType.binary:
                logger.info("Converting image to binary format for binary schema")
                is_success, buffer = cv2.imencode(".jpg", image_data)
                if not is_success:
                    raise ValueError("Failed to encode image")
                    
                # For binary column spec, return a dataframe with the binary data
                import pandas as pd
                return pd.DataFrame({schema.input_names()[0]: [buffer.tobytes()]})
                
            # For tensor input, check if reshaping is needed
            elif schema.is_tensor_spec():
                data_type, data_shape = schema.numpy_types()[0], schema.inputs[0].shape
                logger.info(f"Preparing tensor input with type {data_type} and shape {data_shape}")
                
                # If shape is (-1,), create a batch of 1 with object dtype
                if data_shape == (-1,):
                    batch = np.empty(1, dtype=data_type)
                    batch[0] = image_data
                    return batch
                    
                # If shape requires reshaping, try to reshape
                if len(data_shape) != len(image_data.shape):
                    # Create batch dimension if needed
                    if len(data_shape) == 4 and len(image_data.shape) == 3:
                        return np.expand_dims(image_data.astype(data_type), axis=0)
                    elif len(data_shape) == 2 and len(image_data.shape) == 3:
                        # Model expects 2D input but we have 3D image
                        # Convert to grayscale
                        logger.info("Converting 3D color image to 2D grayscale")
                        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                        return gray.astype(data_type)
                
                # Otherwise, just convert type
                return image_data.astype(data_type)
                
        # Default: return original image
        return image_data
            
    except Exception as e:
        logger.error(f"Error preparing image input: {str(e)}")
        return image_data

def run_generic_model_predict(model, image_data, confidence_threshold=50):
    """
    Run model prediction in the simplest possible way to avoid syntax errors
    
    Args:
        model: MLflow model object
        image_data: Input image data (numpy array)
        confidence_threshold: Confidence threshold for filtering predictions
        
    Returns:
        List of detection dictionaries with standardized format
    """
    try:
        # Just call predict directly - simplest approach from good.py
        predict_result = model.predict(image_data)
        
        # Log the result type for debugging
        logger.info(f"Prediction result type: {type(predict_result)}")
        
        # Initialize empty detections list
        detections = []
        
        # Handle dictionary results (most common for instance segmentation)
        if isinstance(predict_result, dict):
            logger.info(f"Dict result keys: {list(predict_result.keys())}")
            
            # Handle masks and scores format
            if 'masks' in predict_result and 'scores' in predict_result:
                masks = predict_result.get('masks', [])
                scores = predict_result.get('scores', [])
                classes = predict_result.get('classes', [1] * len(scores))
                
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    if score * 100 < confidence_threshold:
                        continue
                        
                    class_id = int(classes[i]) if i < len(classes) else 1
                    
                    if isinstance(mask, np.ndarray):
                        try:
                            # Convert mask to binary
                            binary_mask = (mask > 0.5).astype(np.uint8) if mask.dtype in [np.float32, np.float64] else mask.astype(np.uint8)
                            
                            # Find contours
                            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                segmentation = largest_contour.flatten().tolist()
                                
                                detection = {
                                    "segmentation": segmentation,
                                    "score": float(score),
                                    "class_id": class_id
                                }
                                detections.append(detection)
                        except Exception as e:
                            logger.error(f"Error processing mask: {str(e)}")
            
            # Handle direct segmentation format
            elif 'segmentation' in predict_result:
                segmentations = predict_result.get('segmentation', [])
                scores = predict_result.get('scores', [1.0] * len(segmentations))
                classes = predict_result.get('classes', [1] * len(segmentations))
                
                for i, segmentation in enumerate(segmentations):
                    score = scores[i] if i < len(scores) else 1.0
                    
                    if score * 100 < confidence_threshold:
                        continue
                        
                    class_id = classes[i] if i < len(classes) else 1
                    
                    detection = {
                        "segmentation": segmentation,
                        "score": float(score),
                        "class_id": class_id
                    }
                    detections.append(detection)
            
            # Handle predictions key format
            elif 'predictions' in predict_result:
                predictions = predict_result['predictions']
                
                if isinstance(predictions, list):
                    for pred in predictions:
                        if isinstance(pred, dict) and 'segmentation' in pred:
                            score = pred.get('score', 1.0)
                            
                            if score * 100 < confidence_threshold:
                                continue
                                
                            class_id = pred.get('class_id', 1)
                            
                            detection = {
                                "segmentation": pred['segmentation'],
                                "score": float(score),
                                "class_id": class_id
                            }
                            detections.append(detection)
        
        # Handle list results
        elif isinstance(predict_result, list):
            logger.info(f"List result with {len(predict_result)} items")
            
            for item in predict_result:
                if isinstance(item, dict):
                    # Check for segmentation key
                    if 'segmentation' in item:
                        score = item.get('score', 1.0)
                        
                        if score * 100 < confidence_threshold:
                            continue
                            
                        class_id = item.get('class_id', 1)
                        
                        detection = {
                            "segmentation": item['segmentation'],
                            "score": float(score),
                            "class_id": class_id
                        }
                        detections.append(detection)
                    # Check for mask key
                    elif 'mask' in item or 'masks' in item:
                        mask_key = 'mask' if 'mask' in item else 'masks'
                        mask = item[mask_key]
                        score = item.get('score', 1.0)
                        
                        if score * 100 < confidence_threshold:
                            continue
                            
                        # Process the mask
                        try:
                            if isinstance(mask, np.ndarray):
                                binary_mask = (mask > 0.5).astype(np.uint8) if mask.dtype in [np.float32, np.float64] else mask.astype(np.uint8)
                                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                if contours:
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    segmentation = largest_contour.flatten().tolist()
                                    
                                    detection = {
                                        "segmentation": segmentation,
                                        "score": float(score),
                                        "class_id": item.get('class_id', 1)
                                    }
                                    detections.append(detection)
                        except Exception as e:
                            logger.error(f"Error processing mask: {str(e)}")
        
        # Handle numpy array results
        elif isinstance(predict_result, np.ndarray):
            logger.info(f"Numpy array result with shape {predict_result.shape}")
            
            # Handle masks format
            if len(predict_result.shape) == 3 and predict_result.shape[0] > 0:
                for i in range(predict_result.shape[0]):
                    mask = predict_result[i]
                    
                    try:
                        # Convert mask to binary
                        binary_mask = (mask > 0.5).astype(np.uint8) if mask.dtype in [np.float32, np.float64] else mask.astype(np.uint8)
                        
                        # Find contours
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            segmentation = largest_contour.flatten().tolist()
                            
                            detection = {
                                "segmentation": segmentation,
                                "score": 1.0,  # Default score
                                "class_id": 1   # Default class
                            }
                            detections.append(detection)
                    except Exception as e:
                        logger.error(f"Error processing mask array: {str(e)}")
        
        logger.info(f"Extracted {len(detections)} detections")
        return detections
        
    except Exception as e:
        logger.error(f"Error in run_generic_model_predict: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        return []

def multiply_coordinates(segmentation, scale_x, scale_y):
    """
    Multiply coordinates by scale factors
    
    Args:
        segmentation: Flat list of coordinates [x1, y1, x2, y2, ...]
        scale_x: Scale factor for x coordinates
        scale_y: Scale factor for y coordinates
        
    Returns:
        Scaled coordinates
    """
    if not segmentation or len(segmentation) < 2:
        return segmentation
    
    scaled = []
    for i in range(0, len(segmentation), 2):
        if i+1 < len(segmentation):
            scaled.append(segmentation[i] * scale_x)
            scaled.append(segmentation[i+1] * scale_y)
    
    return scaled

def create_coordinate_pairs(segmentation):
    """Convert flat segmentation list to list of coordinate pairs"""
    if not segmentation or len(segmentation) < 2:
        return []
        
    pairs = []
    for i in range(0, len(segmentation), 2):
        if i+1 < len(segmentation):
            pairs.append((segmentation[i], segmentation[i+1]))
    return pairs

def flatten_coordinates(coords):
    """Convert list of coordinate pairs to flat list"""
    if not coords:
        return []
        
    flat = []
    for x, y in coords:
        flat.extend([x, y])
    return flat

def recalculate_coordinates(segmentation, offset):
    """
    Recalculate segmentation coordinates based on sliding window offset
    
    Args:
        segmentation: Original segmentation coordinates
        offset: (x, y) offset of the sliding window
        
    Returns:
        Recalculated segmentation coordinates
    """
    offset_x, offset_y = offset
    
    # Handle different segmentation formats
    if isinstance(segmentation, list):
        # Handle nested lists (multiple polygons)
        if len(segmentation) > 0 and isinstance(segmentation[0], list):
            return [recalculate_coordinates(poly, offset) for poly in segmentation]
        
        # Single polygon as a flat list of coordinates [x1, y1, x2, y2, ...]
        recalculated = []
        for i in range(0, len(segmentation), 2):
            if i+1 < len(segmentation):
                recalculated.append(segmentation[i] + offset_x)
                recalculated.append(segmentation[i+1] + offset_y)
        return recalculated
    
    # Handle numpy arrays
    elif isinstance(segmentation, np.ndarray):
        if len(segmentation.shape) == 2 and segmentation.shape[1] == 2:
            # Array of [x, y] points
            return segmentation + np.array([offset_x, offset_y])
        else:
            # Try to convert to flat list and process
            return recalculate_coordinates(segmentation.flatten().tolist(), offset)
    
    logger.warning(f"Unsupported segmentation format: {type(segmentation)}")
    return segmentation

def flatten_polygon(polygon):
    """
    Flatten nested polygon structures to a single flat list
    
    Args:
        polygon: Polygon coordinates, possibly nested
        
    Returns:
        Flattened list of coordinates [x1, y1, x2, y2, ...]
    """
    if not polygon:
        return []
        
    # Already flat list with even number of elements
    if isinstance(polygon, list) and len(polygon) % 2 == 0 and all(isinstance(x, (int, float)) for x in polygon):
        return polygon
    
    # Handle numpy arrays
    if isinstance(polygon, np.ndarray):
        if len(polygon.shape) == 2 and polygon.shape[1] == 2:
            # Convert 2D array of points to flat list
            return polygon.flatten().tolist()
        else:
            # Try to convert to flat list
            return flatten_polygon(polygon.tolist())
    
    # Handle nested lists
    if isinstance(polygon, list):
        # Nested structure with one level [[x1, y1], [x2, y2], ...]
        if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in polygon):
            return [coord for point in polygon for coord in point]
        
        # Multiple polygons [[poly1], [poly2], ...]
        if len(polygon) > 0 and isinstance(polygon[0], list):
            # Take the first polygon if there are multiple
            return flatten_polygon(polygon[0])
    
    logger.warning(f"Couldn't flatten polygon: {polygon}")
    return polygon if isinstance(polygon, list) else []

def calculate_bbox_from_segmentation(segmentation):
    """
    Calculate bounding box from segmentation coordinates
    
    Args:
        segmentation: Flat list of coordinates [x1, y1, x2, y2, ...]
        
    Returns:
        Bounding box [x, y, width, height]
    """
    if not segmentation or len(segmentation) < 4:
        return [0, 0, 0, 0]
    
    # Extract x and y coordinates
    x_coords = segmentation[0::2]
    y_coords = segmentation[1::2]
    
    # Calculate bbox
    x_min = min(x_coords)
    y_min = min(y_coords)
    width = max(x_coords) - x_min
    height = max(y_coords) - y_min
    
    return [x_min, y_min, width, height]

def calculate_polygon_area(segmentation):
    """
    Calculate area of a polygon
    
    Args:
        segmentation: Flat list of coordinates [x1, y1, x2, y2, ...]
        
    Returns:
        Area of the polygon
    """
    if not segmentation or len(segmentation) < 6:  # Need at least 3 points
        return 0
    
    # Convert to list of points [(x1, y1), (x2, y2), ...]
    points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
    
    # Calculate area using Shoelace formula
    area = 0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    
    return abs(area) / 2
