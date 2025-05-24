"""
Utility Functions for Parallel Facade AI Inference Component
This module contains essential functions for the parallel component
"""
import os
import io
import sys
import time
import json
import numpy as np
import cv2
import logging
import mlflow
import base64
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model constants for use across the application
class ModelType:
    INSTANCE_SEGMENTATION = "instance_segmentation"
    OBJECT_DETECTION = "object_detection"
    
    # Map model names to category IDs
    MODEL_CLASS = {
        'windows_detection': 2,
        'defective_mastic': 3,
        'StoneFractions': 1,
        'Stonework-Fractures': 2,
        'short_gasket': 1,
        'cleaning_required': 4,
        'mechanical_faults': 5
    }

# Map model names to category IDs (duplicate for backward compatibility)
MODEL_CATEGORY_IDS = {
        'windows_detection': 2,
        'defective_mastic': 3,
        'StoneFractions': 1,
        'Stonework-Fractures': 2,
        'short_gasket': 1,
        'cleaning_required': 4,
        'mechanical_faults': 5
}

# Map model flags to model names
MODEL_NAMES_FROM_FLAGS = {
    'model_windows_detection': 'windows_detection',
    'model_defective_mastic': 'defective_mastic', 
    'model_stone_fractions': 'StoneFractions',
    'model_stonework_fractures': 'Stonework-Fractures',
    'model_short_gasket': 'short_gasket',
    'model_cleaning_required': 'cleaning_required',
    'model_mechanical_faults': 'mechanical_faults'
}

def log_message(message: str) -> None:
    """Log a message to both console and mlflow if available"""
    logger.info(message)
    try:
        mlflow.log_text(message + "\n", "execution_logs.txt")
    except:
        pass  # Ignore mlflow errors

def sliding_window(image: np.ndarray, window_size: int, overlap: int) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Split an image into sliding windows
    
    Args:
        image: Input image as numpy array
        window_size: Size of the sliding window (square)
        overlap: Overlap between windows in pixels
        
    Returns:
        List of tuples containing (window_image, (x, y)) - offset coordinates
    """
    height, width = image.shape[:2]
    windows = []
    
    step = window_size - overlap
    
    for y in range(0, height - overlap, step):
        for x in range(0, width - overlap, step):
            # Adjust window dimensions if it goes beyond image boundaries
            w = min(window_size, width - x)
            h = min(window_size, height - y)
            
            # Skip if window is too small
            if w < window_size // 2 or h < window_size // 2:
                continue
                
            # Extract the window
            window = image[y:y+h, x:x+w]
            
            # Pad if necessary to maintain window_size dimensions
            if w < window_size or h < window_size:
                padded_window = np.zeros((window_size, window_size, 3), dtype=np.uint8)
                padded_window[:h, :w] = window
                window = padded_window
                
            # Only return x, y offset coordinates (not w, h)
            windows.append((window, (x, y)))
    
    log_message(f"Created {len(windows)} sliding windows of size {window_size}x{window_size} with {overlap} overlap")
    return windows

def run_generic_model_predict(model, image_data, confidence_threshold=50):
    """
    Generic model prediction function that handles both numpy arrays and binary inputs.
    
    Args:
        model: MLflow model object
        image_data: Either numpy array or binary image data
        confidence_threshold: Confidence threshold (0-100)
    
    Returns:
        List of detections with segmentation, confidence, etc.
    """
    try:
        # Normalize confidence to 0-1 range if it's in 0-100 range
        conf_threshold = confidence_threshold / 100.0 if confidence_threshold > 1 else confidence_threshold
        
        # Check if image_data is numpy array
        if isinstance(image_data, np.ndarray):
            # Convert to binary format expected by the model
            is_success, buffer = cv2.imencode(".jpg", image_data)
            if not is_success:
                logger.error("Failed to encode image")
                return []
            
            # Convert to binary format
            binary_data = buffer.tobytes()
            
            # Create input with expected schema (binary)
            model_input = {"image": binary_data}
        elif isinstance(image_data, bytes):
            # Already in binary format
            model_input = {"image": image_data}
        else:
            logger.error(f"Unsupported image data type: {type(image_data)}")
            return []
        
        # Run prediction
        start_time = time.time()
        try:
            result = model.predict(model_input)
        except Exception as e:
            # Try alternative formats if first attempt fails
            logger.warning(f"First prediction attempt failed: {str(e)}, trying alternative format")
            try:
                # Try direct predict without dict wrapping
                result = model.predict(binary_data)
            except Exception as e2:
                logger.error(f"Error running model prediction: {str(e2)}")
                return []
                
        inference_time = time.time() - start_time
        logger.info(f"Model inference took {inference_time:.4f} seconds")
        
        # Process results - handle different result formats
        detections = []
        
        # Handle DataFrame output specifically
        import pandas as pd
        if isinstance(result, pd.DataFrame):
            logger.info(f"Model returned DataFrame with columns: {result.columns.tolist()}")
            
            # If the DataFrame has prediction info, extract it
            if 'confidence' in result.columns:
                for _, row in result.iterrows():
                    # Skip if confidence below threshold
                    confidence = row.get('confidence', 0)
                    if confidence < conf_threshold:
                        continue
                        
                    # Create a detection object with a simplified segmentation format
                    detection = {
                        "confidence": confidence
                    }
                    
                    # Extract segmentation if available in some form
                    if 'segmentation' in row:
                        detection["segmentation"] = row['segmentation']
                    elif 'bbox' in row:
                        # Convert bbox to segmentation format if no segmentation
                        x, y, w, h = row['bbox']
                        # Create a rectangle from bbox
                        detection["segmentation"] = [
                            [x, y, x+w, y, x+w, y+h, x, y+h]
                        ]
                    elif all(col in row for col in ['x', 'y', 'width', 'height']):
                        # Convert x,y,width,height to segmentation
                        x, y, w, h = row['x'], row['y'], row['width'], row['height']
                        detection["segmentation"] = [
                            [x, y, x+w, y, x+w, y+h, x, y+h]
                        ]
                    else:
                        # If no segmentation or bbox, create a placeholder
                        detection["segmentation"] = [[0, 0, 10, 0, 10, 10, 0, 10]]
                        
                    # Add class if available
                    if 'class' in row:
                        detection["class"] = row['class']
                    elif 'category_id' in row:
                        detection["class"] = row['category_id']
                    
                    detections.append(detection)
            
            logger.info(f"Extracted {len(detections)} detections from DataFrame")
            return detections
            
        # Parse other result formats
        if isinstance(result, dict) and "predictions" in result:
            # Format: {"predictions": [...]}
            predictions = result["predictions"]
        elif isinstance(result, list):
            # Format: direct list of predictions
            predictions = result
        else:
            logger.warning(f"Unexpected result format: {type(result)}")
            return []
            
        # Process the predictions
        for pred in predictions:
            # Skip if confidence is below threshold
            if "confidence" in pred and pred["confidence"] < conf_threshold:
                continue
                
            # Add to detections
            detections.append(pred)
        
        logger.info(f"Found {len(detections)} detections above confidence threshold {conf_threshold}")
        return detections
        
    except Exception as e:
        logger.error(f"Error running model prediction: {str(e)}")
        return []

def create_coordinate_pairs(coordinates: List[float]) -> List[Tuple[float, float]]:
    """Convert flat list of coordinates to list of (x,y) tuples"""
    return [(coordinates[i], coordinates[i+1]) for i in range(0, len(coordinates), 2)]

def multiply_coordinates(coord_pairs: List[Tuple[float, float]], width: int, height: int) -> List[Tuple[float, float]]:
    """Convert normalized coordinates (0-1) to pixel coordinates"""
    return [(pair[0] * width, pair[1] * height) for pair in coord_pairs]

def flatten_coordinates(coord_pairs: List[Tuple[float, float]]) -> List[float]:
    """Convert list of (x,y) tuples to flat list of coordinates"""
    flat_coords = []
    for pair in coord_pairs:
        flat_coords.extend([pair[0], pair[1]])
    return flat_coords

def flatten_polygon(segmentation: Union[List[List[float]], List[float]]) -> List[float]:
    """Flatten nested segmentation format to a single list of coordinates"""
    if isinstance(segmentation, list):
        if len(segmentation) > 0 and isinstance(segmentation[0], list):
            return segmentation[0]  # Take first polygon if multiple
    return segmentation

def log_inference_metrics(model_name: str, inference_time: float, num_detections: int) -> None:
    """Log inference metrics to MLflow"""
    try:
        mlflow.log_metric(f"{model_name}_inference_time", inference_time)
        mlflow.log_metric(f"{model_name}_detections", num_detections)
        logger.info(f"Logged metrics for {model_name}: inference_time={inference_time:.2f}s, detections={num_detections}")
    except Exception as e:
        logger.error(f"Failed to log metrics: {str(e)}")

def get_cosmos_key_from_keyvault(key_vault_url: str, key_name: str='CosmosDBConnectionString') -> str:
    """Get CosmosDB key from Azure Key Vault
    
    Args:
        key_vault_url: URL of the key vault (e.g., https://myvault.vault.azure.net/)
        key_name: Name of the secret containing CosmosDB key
        
    Returns:
        CosmosDB key as string
    """
    try:
        from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
        from azure.keyvault.secrets import SecretClient
        
        # Try ManagedIdentityCredential first when in Azure environment
        credential = None
        try:
            client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
            logger.info(f"Attempting to use Managed Identity with client ID: {client_id if client_id else 'None (system-assigned)'}")
            
            if client_id:
                credential = ManagedIdentityCredential(client_id=client_id)
            else:
                credential = ManagedIdentityCredential()
            
            # Test the credential by trying to get token
            token = credential.get_token("https://vault.azure.net/.default")
            logger.info("Successfully obtained token using ManagedIdentityCredential for KeyVault")
        except Exception as e:
            logger.warning(f"Failed to use ManagedIdentityCredential for KeyVault: {str(e)}")
            logger.info("Falling back to DefaultAzureCredential for KeyVault")
            credential = DefaultAzureCredential()
        
        client = SecretClient(vault_url=key_vault_url, credential=credential)
        secret = client.get_secret(key_name)
        logger.info(f"Successfully retrieved secret '{key_name}' from KeyVault")
        return secret.value
    except Exception as e:
        logger.error(f"Error accessing key vault at {key_vault_url}: {str(e)}")
        return None

def recalculate_coordinates(coords: List[float], window_offset: Tuple[int, int]) -> List[float]:
    """
    Recalculate polygon coordinates based on sliding window offset
    
    Args:
        coords: Flat list of coordinates [x1, y1, x2, y2, ...]
        window_offset: (x, y) offset of the sliding window
        
    Returns:
        List of adjusted coordinates
    """
    x_offset, y_offset = window_offset
    new_coords = []
    
    for i in range(0, len(coords), 2):
        if i+1 < len(coords):
            new_coords.append(coords[i] + x_offset)
            new_coords.append(coords[i+1] + y_offset)
            
    return new_coords

def calculate_bbox_from_segmentation(segmentation: List[float]) -> List[float]:
    """
    Calculate bounding box from segmentation polygon
    
    Args:
        segmentation: Flat list of coordinates [x1, y1, x2, y2, ...]
        
    Returns:
        Bounding box as [x, y, width, height]
    """
    x_coords = segmentation[0::2]
    y_coords = segmentation[1::2]
    
    if not x_coords or not y_coords:
        return [0, 0, 0, 0]
        
    x_min = min(x_coords)
    y_min = min(y_coords)
    width = max(x_coords) - x_min
    height = max(y_coords) - y_min
    
    return [x_min, y_min, width, height]

def calculate_polygon_area(segmentation: List[float]) -> float:
    """
    Calculate area of a polygon from segmentation coordinates
    
    Args:
        segmentation: Flat list of coordinates [x1, y1, x2, y2, ...]
        
    Returns:
        Area of the polygon
    """
    # Convert flat list to pairs of coordinates
    points = []
    for i in range(0, len(segmentation), 2):
        if i+1 < len(segmentation):
            points.append((segmentation[i], segmentation[i+1]))
    
    # Calculate area using Shoelace formula
    area = 0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2
    
    return area

def preprocess_image_for_model(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for model input
    
    Args:
        image: Input image in BGR format (OpenCV default)
        
    Returns:
        Preprocessed image in RGB format
    """
    # Convert BGR to RGB for model input
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # If not a 3-channel color image, just pass through
        rgb_image = image
        
    return rgb_image

def get_cosmosdb_client(connection_string: str = None, key_vault_url: str = None):
    """
    Get Azure CosmosDB client
    
    Args:
        connection_string: CosmosDB connection string
        key_vault_url: URL to Azure Key Vault
        
    Returns:
        CosmosDB client
    """
    try:
        from azure.cosmos import CosmosClient
        
        if connection_string:
            # Parse connection string for endpoint and key
            if ';' in connection_string:
                params = dict(item.split('=', 1) for item in connection_string.split(';') if '=' in item)
                endpoint = params.get('AccountEndpoint')
                key = params.get('AccountKey')
                
                if not endpoint or not key:
                    logger.error("Invalid CosmosDB connection string format")
                    return None
                    
                client = CosmosClient(endpoint, key)
            else:
                # Try direct connection string format
                client = CosmosClient.from_connection_string(connection_string)
                
            logger.info("Successfully connected to CosmosDB")
            return client
        elif key_vault_url:
            # Get connection string from Key Vault
            cosmos_key = get_cosmos_key_from_keyvault(key_vault_url)
            if cosmos_key:
                return get_cosmosdb_client(cosmos_key)
            else:
                logger.error("Failed to get CosmosDB key from Key Vault")
                return None
        else:
            logger.error("No CosmosDB connection string or Key Vault URL provided")
            return None
    except Exception as e:
        logger.error(f"Error connecting to CosmosDB: {str(e)}")
        return None

def get_processed_ids(cosmos_client, database_id: str, container_id: str, batch_id: str = None):
    """
    Get list of processed image IDs from CosmosDB
    
    Args:
        cosmos_client: CosmosDB client
        database_id: Database ID
        container_id: Container ID
        batch_id: Optional batch ID to filter by
        
    Returns:
        List of processed image IDs
    """
    try:
        database = cosmos_client.get_database_client(database_id)
        container = database.get_container_client(container_id)
        
        # Query for processed images (status not empty)
        if batch_id:
            query = f"SELECT c.id FROM c WHERE c.BatchID = '{batch_id}' AND c.Status != ''"
        else:
            query = "SELECT c.id FROM c WHERE c.Status != ''"
            
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        processed_ids = [item['id'] for item in items]
        logger.info(f"Found {len(processed_ids)} processed image IDs")
        return processed_ids
    except Exception as e:
        logger.error(f"Error getting processed IDs: {str(e)}")
        return []

def update_cosmos_db(cosmos_client, image_id: str, annotations: List[Dict], 
                    batch_id: str = None, database_name: str = "FacadeDB", 
                    container_name: str = "Images"):
    """
    Update CosmosDB record with new annotations
    
    Args:
        cosmos_client: CosmosDB client
        image_id: Image ID
        annotations: List of annotations
        batch_id: Batch ID
        database_name: Database name
        container_name: Container name
        
    Returns:
        Boolean success status
    """
    try:
        database = cosmos_client.get_database_client(database_name)
        container = database.get_container_client(container_name)
        
        # Get current record
        try:
            record = container.read_item(item=image_id, partition_key=image_id)
            logger.info(f"Retrieved existing record for image {image_id}")
        except Exception as e:
            logger.info(f"No existing record found for {image_id}, creating new record")
            # Create new record
            record = {
                "id": image_id,
                "ImageID": image_id,
                "BatchID": batch_id or "unknown",
                "Status": "",
                "info": {
                    "description": "Facade Studio Annotations",
                    "date_created": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())
                },
                "images": [
                    {
                        "id": image_id,
                        "file_name": f"{batch_id or 'unknown'}/cam/{image_id}.jpg",
                        "width": 3840,  # Default width
                        "height": 2160  # Default height
                    }
                ],
                "annotations": [],
                "categories": [
                    {"id": 1, "name": "Short-Gasket", "supercategory": "Facade"},
                    {"id": 2, "name": "Stonework-Fracture", "supercategory": "Facade"},
                    {"id": 3, "name": "Defective-Mastic", "supercategory": "Facade"},
                    {"id": 4, "name": "Cleaning-Required", "supercategory": "Facade"},
                    {"id": 5, "name": "Mechanical-Fault", "supercategory": "Facade"}
                ]
            }
        
        # Update or append annotations
        if 'annotations' not in record:
            record['annotations'] = []
            
        # Find max ID for existing annotations
        next_id = 1
        if record['annotations']:
            next_id = max(ann.get('id', 0) for ann in record['annotations']) + 1
            
        # Add annotations with sequential IDs
        for annotation in annotations:
            annotation['id'] = next_id
            annotation['image_id'] = image_id
            annotation['objectId'] = annotation.get('objectId', f"{int(time.time() * 1000)}")
            record['annotations'].append(annotation)
            next_id += 1
            
        # Update status
        record['Status'] = 'Processed'
        
        # Update record
        container.upsert_item(body=record)
        logger.info(f"Updated record {image_id} with {len(annotations)} annotations")
        return True
    except Exception as e:
        logger.error(f"Error updating cosmos record {image_id}: {str(e)}")
        return False

def save_local_results(output_dir: str, image_id: str, annotations: List[Dict]) -> bool:
    """Save detection results locally as JSON file"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{image_id}_annotations.json")
        
        # Create a simplified record structure
        record = {
            "id": image_id,
            "annotations": annotations,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())
        }
        
        with open(output_file, 'w') as f:
            json.dump(record, f, indent=2)
            
        logger.info(f"Saved local results for {image_id} to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving local results for {image_id}: {str(e)}")
        return False

def encode_image_to_binary(image: np.ndarray) -> bytes:
    """
    Encode image to binary format
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Binary representation of the image
    """
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode image")
    return encoded_image.tobytes()

def run_window_detection_model(model, image, confidence_threshold=50):
    """Run the window detection model on an image"""
    return run_generic_model_predict(model, image, confidence_threshold)
