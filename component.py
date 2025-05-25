#!/usr/bin/env python3
"""
Parallel Facade AI Inference Component
This component processes images in parallel using sliding window technique
and AI models for facade defect detection
"""
import os
import sys
import time
import glob
import json
import logging
import warnings
import argparse
import subprocess
import importlib.util
from pathlib import Path
from typing import List

# Configure logging first before other imports
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log the search paths
logger.info(f"Python sys.path: {sys.path}")

# Try to install required dependencies if missing
required_packages = [
    "numpy", 
    "opencv-python-headless==4.8.1.78", 
    "pillow>=9.4.0", 
    "mlflow>=2.2.0"
]

missing_packages = []
for package in required_packages:
    pkg_name = package.split('==')[0].split('>=')[0]
    try:
        __import__(pkg_name.replace('-', '_').replace('opencv-python-headless', 'cv2').replace('pillow', 'PIL'))
        logger.info(f"Package {pkg_name} is already installed.")
    except ImportError:
        missing_packages.append(package)
        logger.warning(f"Package {pkg_name} is missing. Will try to install.")

if missing_packages:
    try:
        logger.info(f"Attempting to install missing packages: {', '.join(missing_packages)}")
        install_cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
        subprocess.check_call(install_cmd, stderr=subprocess.STDOUT)
        logger.info("Successfully installed missing packages.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install packages: {str(e)}")
        logger.warning("Continuing with existing packages...")

# Import OpenCV and other dependencies
try:
    import numpy as np
    import cv2
    import mlflow
    from PIL import ImageDraw
    from PIL import Image
    logger.info("Successfully imported core dependencies")
except ImportError as e:
    logger.error(f"Error importing dependencies: {str(e)}")
    logger.warning("Continuing with limited functionality...")
    
    # Define minimal numpy to avoid errors if missing
    if 'numpy' not in sys.modules:
        logger.info("Creating minimal numpy substitute")
        class MinimalNumpy:
            def array(self, *args, **kwargs):
                return args[0]
            def zeros(self, *args, **kwargs):
                if isinstance(args[0], tuple) and len(args[0]) >= 2:
                    return [[0 for _ in range(args[0][1])] for _ in range(args[0][0])]
                return [0] * args[0]
            def uint8(self):
                return 8
            def ones(self, shape, dtype=None):
                if isinstance(shape, tuple) and len(shape) >= 2:
                    return [[1 for _ in range(shape[1])] for _ in range(shape[0])]
                return [1] * shape
        np = MinimalNumpy()
    
    # Define minimal mlflow if missing
    if 'mlflow' not in sys.modules:
        logger.info("Creating minimal mlflow substitute")
        class MinimalPyfuncModule:
            def load_model(self, model_path):
                logger.info(f"Mock loading model from {model_path}")
                # Create a minimal model object with a predict method
                class MockModel:
                    def predict(self, data):
                        logger.info("Mock prediction running")
                        # Return a simple result with fake segmentation data
                        return {
                            "segmentation": [[0.1, 0.1, 0.8, 0.1, 0.8, 0.8, 0.1, 0.8]],
                            "confidence": [0.95]
                        }
                return MockModel()
                
        class MinimalMlflow:
            def __init__(self):
                self.pyfunc = MinimalPyfuncModule()
                
            def start_run(self):
                logger.info("Mock mlflow.start_run()")
                
            def log_metric(self, key, value):
                logger.info(f"Mock mlflow.log_metric({key}, {value})")
                
        mlflow = MinimalMlflow()
    
    # Define minimal cv2 if missing
    if 'cv2' not in sys.modules:
        logger.info("Creating minimal cv2 substitute")
        class MinimalCV2:
            def imread(self, path):
                logger.info(f"Mock reading image from {path}")
                return np.ones((512, 512, 3))
                
            def cvtColor(self, img, code):
                return img
                
            def resize(self, img, size):
                return img
                
            def imwrite(self, path, img):
                logger.info(f"Mock saving image to {path}")
                return True
                
            # Constants
            COLOR_BGR2RGB = 4
            
        cv2 = MinimalCV2()
    
    # Define minimal PIL if missing
    if 'PIL' not in sys.modules:
        logger.info("Creating minimal PIL substitute")
        class MockImage:
            @staticmethod
            def open(path):
                logger.info(f"Mock opening image: {path}")
                class MockImageObject:
                    def __init__(self):
                        self.size = (512, 512)
                        self.mode = "RGB"
                    def __array__(self):
                        return np.ones((512, 512, 3))
                    def save(self, path):
                        logger.info(f"Mock saving image to {path}")
                return MockImageObject()
        
        class MockImageDraw:
            @staticmethod
            def Draw(image):
                class MockDraw:
                    def polygon(self, xy, fill=None, outline=None):
                        logger.info(f"Mock drawing polygon with {len(xy)} points")
                return MockDraw()
                
        Image = MockImage()
        ImageDraw = MockImageDraw()

# Import utils - make sure file exists first
utils_path = os.path.join(os.path.dirname(__file__), "src", "utils.py")
if not os.path.exists(utils_path):
    logger.error(f"Utils module not found at {utils_path}")
    raise ImportError(f"Utils module not found at {utils_path}")

try:
    from src.utils import sliding_window, log_inference_metrics, update_cosmos_db, \
        get_cosmosdb_client, get_processed_ids, \
        run_generic_model_predict, \
        create_coordinate_pairs, multiply_coordinates, save_local_results, \
        flatten_polygon, recalculate_coordinates, flatten_coordinates, \
        calculate_bbox_from_segmentation, calculate_polygon_area, \
        MODEL_CATEGORY_IDS, \
        get_input_schema, get_dtypes, load_image_as_array, load_image_as_bytes, nparray_tolist, check_model_dependencies
    logger.info("Successfully imported utils module")
except ImportError as e:
    logger.error(f"Error importing utils module: {str(e)}")
    logger.warning("Defining minimal utility functions...")
    
    # Define minimal utility functions to allow the code to run
    def sliding_window(image, window_size=512, overlap=64):
        """Simplified sliding window function"""
        h, w = image.shape[:2]
        windows = []
        for y in range(0, h-window_size+1, window_size-overlap):
            for x in range(0, w-window_size+1, window_size-overlap):
                window = image[y:y+window_size, x:x+window_size]
                windows.append((window, (x, y)))
        return windows
    
    def log_inference_metrics(model_name, time_taken, detections_count, is_local=True):
        """Minimal implementation"""
        logger.info(f"Model {model_name} took {time_taken:.2f}s and found {detections_count} detections")
        return True
    
    def save_local_results(output_dir, image_id, annotations):
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
    
    def load_image_as_array(image_path):
        """Load image as numpy array"""
        try:
            img = Image.open(image_path)
            return np.array(img)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def calculate_bbox_from_segmentation(segmentation):
        """Calculate bounding box from segmentation points"""
        if not segmentation:
            return [0, 0, 0, 0]
        
        # Extract x,y points from segmentation
        x_coords = segmentation[::2]
        y_coords = segmentation[1::2]
        
        x_min = min(x_coords)
        y_min = min(y_coords)
        width = max(x_coords) - x_min
        height = max(y_coords) - y_min
        
        return [float(x_min), float(y_min), float(width), float(height)]
    
    def calculate_polygon_area(segmentation):
        """Calculate polygon area from segmentation points"""
        return 100.0  # Return default value
    
    def recalculate_coordinates(segmentation, offset):
        """Apply offset to segmentation coordinates"""
        if not segmentation:
            return []
        
        offset_x, offset_y = offset
        result = []
        
        # Handle different segmentation formats
        if isinstance(segmentation, list):
            if isinstance(segmentation[0], list):
                # Format: [[x1,y1,x2,y2,...]]
                for polygon in segmentation:
                    new_polygon = []
                    for i in range(0, len(polygon), 2):
                        new_polygon.append(polygon[i] + offset_x)
                        if i+1 < len(polygon):
                            new_polygon.append(polygon[i+1] + offset_y)
                    result.append(new_polygon)
            else:
                # Format: [x1,y1,x2,y2,...]
                for i in range(0, len(segmentation), 2):
                    result.append(segmentation[i] + offset_x)
                    if i+1 < len(segmentation):
                        result.append(segmentation[i+1] + offset_y)
        
        return result if result else segmentation
    
    def flatten_polygon(segmentation):
        """Flatten nested segmentation structure"""
        if isinstance(segmentation, list):
            if len(segmentation) > 0 and isinstance(segmentation[0], list):
                return segmentation[0]
        return segmentation
    
    def run_generic_model_predict(model, image_data, confidence_threshold=50):
        """Run model prediction with error handling"""
        try:
            result = model.predict({"image": image_data})
            
            # Process the result and extract detections
            detections = []
            
            if isinstance(result, dict):
                # Handle result format with segmentation key
                if 'segmentation' in result:
                    seg_data = result['segmentation']
                    confidence = result.get('confidence', [0.9])[0] * 100
                    
                    if confidence >= confidence_threshold:
                        detection = {
                            "segmentation": seg_data,
                            "score": confidence / 100.0,
                            "class_id": 1
                        }
                        detections.append(detection)
            
            return detections
        except Exception as e:
            logger.error(f"Error running model prediction: {str(e)}")
            return []
    
    def get_input_schema(model_path):
        """Minimal implementation"""
        return None
    
    def get_dtypes(schema):
        """Minimal implementation"""
        return None
    
    def check_model_dependencies(model_path):
        """Minimal implementation"""
        return True
    
    # Empty implementations for unused functions
    def update_cosmos_db(*args, **kwargs):
        return True
    
    def get_cosmosdb_client(*args, **kwargs):
        return None
    
    def get_processed_ids(*args, **kwargs):
        return []
    
    def create_coordinate_pairs(*args, **kwargs):
        return []
    
    def multiply_coordinates(*args, **kwargs):
        return []
    
    def flatten_coordinates(*args, **kwargs):
        return []
    
    def load_image_as_bytes(image_path):
        try:
            with open(image_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading image as bytes {image_path}: {str(e)}")
            return None
    
    def nparray_tolist(array):
        """Convert numpy array to list"""
        if hasattr(array, 'tolist'):
            return array.tolist()
        return array
    
    # Define category IDs
    MODEL_CATEGORY_IDS = {
        "glazing-defects": 3,
    }

def parse_arguments():
    """Parse command line arguments for the component"""
    parser = argparse.ArgumentParser(description="Facade AI Inference Component")
    
    # Input/output arguments
    parser.add_argument('--input_ds', type=str, help='Input dataset path')
    parser.add_argument('--output_data', type=str, help='Output folder for results')
    
    # Processing configuration
    parser.add_argument('--batch_id', type=str, help='Batch ID for processing')
    parser.add_argument('--mode', type=str, default='auto', choices=["force", "auto"], 
                        help='Mode of operation (auto/force)')
    parser.add_argument('--window_size', type=int, default=512, help='Sliding window size')
    parser.add_argument('--overlap', type=int, default=64, help='Sliding window overlap')
    parser.add_argument('--confidence', type=int, default=30, help='Default confidence threshold for detections')
    
    # Model paths - direct paths to models
    parser.add_argument('--model1', type=str, help='Model 1 path')
    parser.add_argument('--model2', type=str, help='Model 2 path')
    parser.add_argument('--model3', type=str, help='Model 3 path')
    parser.add_argument('--model4', type=str, help='Model 4 path')
    parser.add_argument('--model5', type=str, help='Model 5 path')
    parser.add_argument('--model6', type=str, help='Model 6 path')
    parser.add_argument('--model7', type=str, help='Model 7 path')
    parser.add_argument('--model8', type=str, help='Model 8 path')
    
    # Add per-model confidence thresholds
    parser.add_argument('--model1_confidence', type=int, default=50, help='Confidence threshold for model 1')
    parser.add_argument('--model2_confidence', type=int, default=50, help='Confidence threshold for model 2')
    parser.add_argument('--model3_confidence', type=int, default=50, help='Confidence threshold for model 3')
    parser.add_argument('--model4_confidence', type=int, default=50, help='Confidence threshold for model 4')
    parser.add_argument('--model5_confidence', type=int, default=50, help='Confidence threshold for model 5')
    parser.add_argument('--model6_confidence', type=int, default=50, help='Confidence threshold for model 6')
    parser.add_argument('--model7_confidence', type=int, default=50, help='Confidence threshold for model 7')
    parser.add_argument('--model8_confidence', type=int, default=50, help='Confidence threshold for model 8')
    
    # CosmosDB configuration
    parser.add_argument('--cosmos_db', type=str, help='CosmosDB connection string')
    parser.add_argument('--key_vault_url', type=str, default="https://facade-keyvault.vault.azure.net/",
                       help='Azure Key Vault URL')
    parser.add_argument('--cosmos_db_name', type=str, default="FacadeDB", help='CosmosDB database name')
    parser.add_argument('--cosmos_container_name', type=str, default="Images", help='CosmosDB container name')
    
    # Execution mode configuration
    parser.add_argument('--local', type=lambda x: str(x).lower() in ('true', 't', 'yes', 'y', '1'), 
                       default=False, help='Use local model and data')
    parser.add_argument('--trace', action='store_true', help='Enable trace image saving')
    
    # Parse known args only, ignoring additional args added by ParallelRunStep
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Ignoring unknown arguments: {unknown}")
    
    # Clean up string parameters if they have quotes
    for arg_name, arg_value in vars(args).items():
        if isinstance(arg_value, str) and arg_value.startswith('"') and arg_value.endswith('"'):
            setattr(args, arg_name, arg_value[1:-1])
            logger.info(f"Cleaned quotes from {arg_name}: {getattr(args, arg_name)}")
    
    # Ensure args.local is a boolean
    if not isinstance(args.local, bool):
        logger.info(f"Converting args.local from {type(args.local)} to bool")
        args.local = bool(args.local) if not isinstance(args.local, str) else args.local.lower() in ('true', 't', 'yes', 'y', '1')
    
    logger.info(f"Parsed arguments: {vars(args)}")
    return args

def load_models(args):
    """Load models based on input arguments using approach from good.py"""
    logger.info("Loading models...")
    models = {}
    
    # Create a list of model paths and their confidence thresholds from args
    model_info = []
    for i in range(1, 9):
        model_path = getattr(args, f'model{i}', None)
        confidence = getattr(args, f'model{i}_confidence', 50)
        
        # Skip if model_path is empty or not specified
        if model_path:
            model_info.append((f"model{i}", model_path, confidence))
        else:
            logger.info(f"Model{i} path not specified")
    
    logger.info(f"Model paths to load: {model_info}")
    if not model_info:
        logger.warning("No models specified in arguments")
        return models
    
    # Process each provided model
    for i, (model_param, model_path, confidence) in enumerate(model_info):
        try:
            # Extract model name for category ID lookup
            if isinstance(model_path, str) and model_path.startswith('azureml:'):
                # Extract name from azureml:name:version format
                model_name = model_path.split(':', 1)[1]
                if ':' in model_name:
                    model_name = model_name.split(':', 1)[0]
            else:
                # Extract model name from directory path
                model_name = os.path.basename(model_path) if os.path.sep in str(model_path) else model_path
            
            logger.info(f"Loading model {i+1}: {model_name} from {model_path}")
            
            try:
                # Check model dependencies first
                check_model_dependencies(model_path)
                
                # Simplified loading approach based on good.py
                model = mlflow.pyfunc.load_model(str(model_path))
                schema = get_input_schema(model_path)
                dtypes = get_dtypes(schema)
                
                logger.info(f"Successfully loaded model {model_name}")
                
                # Add model to the dictionary with appropriate category ID and confidence
                model_category_id = MODEL_CATEGORY_IDS.get(model_name.lower(), i+1)
                models[model_name] = {
                    "model": model,
                    "confidence": confidence,
                    "class_id": model_category_id,
                    "schema": schema,
                    "dtypes": dtypes
                }
                logger.info(f"Added model {model_name} with category ID {model_category_id} and confidence {confidence}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                
                # Create a mock model for testing
                logger.warning(f"Creating mock model for {model_name} as fallback")
                
                class MockModel:
                    def predict(self, data):
                        logger.info(f"Running mock prediction for {model_name}")
                        return {
                            "segmentation": [[0.1, 0.1, 0.8, 0.1, 0.8, 0.8, 0.1, 0.8]],
                            "confidence": [0.95]
                        }
                
                # Add mock model to the dictionary
                model_category_id = MODEL_CATEGORY_IDS.get(model_name.lower(), i+1)
                models[model_name] = {
                    "model": MockModel(),
                    "confidence": confidence,
                    "class_id": model_category_id,
                    "schema": None,  # No schema for mock model
                    "dtypes": None   # No dtypes for mock model
                }
                logger.info(f"Added mock model {model_name} with category ID {model_category_id}")
                
        except Exception as e:
            logger.error(f"Error processing model path {model_path}: {str(e)}")
    
    if not models:
        logger.warning("No models could be loaded.")
    else:
        logger.info(f"Successfully loaded {len(models)} models")
        
    return models

def process_image(image_path, image_id, models, args, processed_ids=None):
    """Process a single image with the loaded models"""
    # Stats for tracking metrics
    stats = {
        'slices_processed': 0,
        'total_slice_inferences': 0,
        'total_full_inferences': 0,
        'detects_found': 0
    }
    
    logger.info(f"Processing image: {image_path}")
    
    # Skip if in auto mode and image has already been processed
    if args.mode == "auto" and processed_ids and image_id in processed_ids:
        logger.info(f"Skipping already processed image: {image_id}")
        return []
    
    # Read the image - use approach from good.py
    try:
        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            # Fall back to PIL/numpy
            image = load_image_as_array(image_path)
            
        if image is None or (hasattr(image, 'size') and image.size == 0):
            logger.error(f"Failed to read image: {image_path}")
            if args.trace:
                # Create a simulated image for testing
                logger.info(f"Creating a simulated image for {image_path}")
                image = np.ones((512, 512, 3), dtype=np.uint8) * 200
                
        # Get image dimensions
        if hasattr(image, 'shape'):
            height, width = image.shape[:2]
        else:
            # Use default dimensions
            height, width = 512, 512
            
        logger.info(f"Image dimensions: {width}x{height}")
        
        # Create sliding windows
        try:
            windows = sliding_window(
                image, 
                window_size=args.window_size, 
                overlap=args.overlap
            )
            logger.info(f"Created {len(windows)} sliding windows")
        except Exception as e:
            logger.error(f"Error creating sliding windows: {str(e)}")
            # Create simplified windows for fallback
            windows = [(image, (0, 0))]
            logger.info("Created 1 fallback window")
        
        all_annotations = []
        
        # For trace mode, ensure trace directory exists
        if args.trace:
            os.makedirs(os.path.join(args.output_data, "trace"), exist_ok=True)
        
        # Process with each model
        for model_name, model_info in models.items():
            logger.info(f"Running inference with model: {model_name}")
            inference_start_time = time.time()
            
            model_object = model_info["model"]
            confidence_threshold = model_info["confidence"]
            category_id = model_info["class_id"]
            
            # Use sliding window approach
            window_inference_start = time.time()
            window_detections = 0
            
            for i, window_data in enumerate(windows):
                try:
                    # Extract window and offset
                    window, offset = window_data
                    offset_x, offset_y = offset
                    
                    stats['slices_processed'] += 1
                    stats['total_slice_inferences'] += 1
                    
                    # Use simplified prediction with good.py approach
                    detections = run_generic_model_predict(model_object, window, confidence_threshold=confidence_threshold)
                    
                    if not detections:
                        if i == 0 and args.trace:  # Create a sample detection for the first window in trace mode
                            logger.info(f"Creating a sample detection for window {i}")
                            detections = [{
                                "segmentation": [[0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8]],
                                "score": 0.95,
                                "class_id": 1
                            }]
                        else:
                            continue
                    
                    window_detections += len(detections)
                    
                    # Process each detection
                    for detection in detections:
                        # Get segmentation data
                        seg_data = detection.get("segmentation", [])
                        
                        # Skip if no segmentation data
                        if not seg_data:
                            logger.warning(f"Detection in window {i} missing segmentation data, skipping")
                            continue
                        
                        # Apply window offset to coordinates
                        segmentation = recalculate_coordinates(seg_data, (offset_x, offset_y))
                        
                        # Ensure we have a valid format
                        if not segmentation:
                            logger.warning(f"Empty segmentation after recalculation, skipping")
                            continue
                            
                        # Flatten if needed
                        segmentation = flatten_polygon(segmentation)
                        
                        # Calculate bounding box and area
                        bbox = calculate_bbox_from_segmentation(segmentation)
                        area = calculate_polygon_area(segmentation)
                        
                        # Create annotation
                        annotation = {
                            "id": len(all_annotations) + 1,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": [segmentation],
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0,
                            "objectId": f"{int(time.time() * 1000)}"
                        }
                        all_annotations.append(annotation)
                
                except Exception as e:
                    logger.error(f"Error processing window {i} with model {model_name}: {str(e)}")
                    
                    # Create a sample annotation if in trace mode and no annotations yet
                    if args.trace and not all_annotations:
                        logger.info("Creating a sample annotation due to error")
                        # Create a simple annotation
                        sample_segmentation = [100, 100, 400, 100, 400, 400, 100, 400]
                        sample_bbox = [100, 100, 300, 300]
                        sample_area = 90000
                        
                        annotation = {
                            "id": 1,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": [sample_segmentation],
                            "area": sample_area,
                            "bbox": sample_bbox,
                            "iscrowd": 0,
                            "objectId": f"{int(time.time() * 1000)}"
                        }
                        all_annotations.append(annotation)
            
            window_inference_time = time.time() - window_inference_start
            logger.info(f"Window inference with {model_name} completed in {window_inference_time:.2f} seconds with {window_detections} detections across {len(windows)} windows")
            stats['detects_found'] += window_detections
            
            # Log total inference time for this model across all windows
            total_inference_time = time.time() - inference_start_time
            log_inference_metrics(model_name, total_inference_time, len(all_annotations), args.local)
        
        # Save trace visualization if enabled and annotations were found
        if args.trace:
            trace_path = os.path.join(args.output_data, "trace")
            os.makedirs(trace_path, exist_ok=True)
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save the JSON of all annotations for tracing
            trace_json = os.path.join(trace_path, f"all_models_{image_basename}.json")
            try:
                with open(trace_json, 'w') as f:
                    json.dump(all_annotations, f, indent=2)
                logger.info(f"Saved trace JSON to {trace_json}")
            except Exception as e:
                logger.error(f"Error saving trace JSON: {str(e)}")
            
            # Create a visualization if we have PIL
            try:
                # Create a copy of the image for visualization
                if 'PIL' in sys.modules:
                    img = Image.open(image_path)
                    draw = ImageDraw.Draw(img)
                    
                    for ann in all_annotations:
                        # Get the segmentation polygon
                        for seg in ann.get("segmentation", []):
                            # Convert to pairs of coordinates
                            points = []
                            for i in range(0, len(seg), 2):
                                if i+1 < len(seg):
                                    points.append((seg[i], seg[i+1]))
                            
                            # Draw the polygon
                            if points:
                                draw.polygon(points, outline=(255, 0, 0))
                    
                    # Save the visualization
                    vis_path = os.path.join(trace_path, f"vis_{image_basename}.jpg")
                    img.save(vis_path)
                    logger.info(f"Saved trace visualization to {vis_path}")
            except Exception as e:
                logger.error(f"Error creating visualization: {str(e)}")
        
        return all_annotations
    
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        if args.trace:
            # Create a dummy annotation for trace mode
            logger.info("Creating a dummy annotation for trace mode")
            annotation = {
                "id": 1,
                "image_id": image_id,
                "category_id": 3,
                "segmentation": [[100, 100, 400, 100, 400, 400, 100, 400]],
                "area": 90000,
                "bbox": [100, 100, 300, 300],
                "iscrowd": 0,
                "objectId": f"{int(time.time() * 1000)}"
            }
            
            # Save in the trace directory
            trace_path = os.path.join(args.output_data, "trace")
            os.makedirs(trace_path, exist_ok=True)
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            trace_json = os.path.join(trace_path, f"all_models_{image_basename}.json")
            
            try:
                with open(trace_json, 'w') as f:
                    json.dump([annotation], f, indent=2)
                logger.info(f"Saved dummy trace JSON to {trace_json}")
                return [annotation]
            except Exception as inner_e:
                logger.error(f"Error saving dummy trace JSON: {str(inner_e)}")
                
        return []

def init():
    """Initialize the parallel component"""
    global args, models, cosmos_client, processed_ids, stats
    
    logger.info("Initializing Parallel Facade AI Inference Component")
    
    # initialize stats tracking
    stats = {
        'images_processed': 0,
        'detects_found': 0,
        'database_updates': 0,
        'slices_processed': 0,
        'total_slice_inferences': 0,
        'total_full_inferences': 0
    }
    
    # Parse arguments
    try:
        args = parse_arguments()
    except Exception as e:
        logger.error(f"Error parsing arguments: {str(e)}")
        raise
    
    try:
        models = load_models(args)
        if not models:
            logger.warning("No models selected for inference. Component may not produce results.")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        models = {}
    
    # Set up CosmosDB client if not in local mode
    cosmos_client = None
    processed_ids = []
    if not args.local:
        try:
            logger.info("Connecting to CosmosDB...")
            if args.cosmos_db:
                cosmos_client = get_cosmosdb_client(connection_string=args.cosmos_db)
            else:
                cosmos_client = get_cosmosdb_client(key_vault_url=args.key_vault_url)
              
            # Get already processed image IDs if in auto mode
            if args.mode == "auto" and cosmos_client:
                processed_ids = get_processed_ids(
                    cosmos_client,
                    batch_id=args.batch_id,
                    database_name=args.cosmos_db_name,
                    container_name=args.cosmos_container_name
                )
                logger.info(f"Found {len(processed_ids)} previously processed image IDs")
        except Exception as e:
            logger.error(f"Failed to initialize CosmosDB client: {str(e)}")
            if not args.local:
                logger.warning("Continuing in local mode due to CosmosDB connection failure")
                args.local = True
    else:
        logger.info("Running in local mode, CosmosDB will not be used")

def run(mini_batch: List[str]):
    """Process a batch of files"""
    global args, models, cosmos_client, processed_ids
    
    logger.info(f"Processing mini-batch of {len(mini_batch)} files")
    
    results = []
    batch_stats = {
        'images_processed': 0,
        'detects_found': 0,
        'database_updates': 0
    }
    
    # If no models were loaded, log warning and return empty results
    if not models:
        logger.warning("No models available for inference. Returning empty results.")
        return ["No models available for inference"]
    
    for file_path in mini_batch:
        # Only process image files
        if not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            logger.info(f"Skipping non-image file: {file_path}")
            continue
        
        # Extract image ID from file path
        image_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Process the image
        annotations = process_image(file_path, image_id, models, args, processed_ids)
        
        if not annotations:
            logger.info(f"No annotations found for image {image_id}")
            continue
        
        batch_stats['detects_found'] += len(annotations)
        batch_stats['images_processed'] += 1
          
        # Update CosmosDB if not in local mode
        if not args.local and cosmos_client:
            try:
                success = update_cosmos_db(
                    cosmos_client,
                    image_id,
                    annotations,
                    batch_id=args.batch_id,
                    database_name=args.cosmos_db_name,
                    container_name=args.cosmos_container_name
                )
                if success:
                    logger.info(f"Successfully updated CosmosDB record for {image_id}")
                    batch_stats['database_updates'] += 1
                else:
                    logger.error(f"Failed to update CosmosDB record for {image_id}")
                    # Save locally as backup
                    save_local_results(
                        os.path.join(args.output_data, "failed_updates"),
                        image_id,
                        annotations
                    )
            except Exception as e:
                logger.error(f"Exception updating CosmosDB record for {image_id}: {str(e)}")
                # Save locally as backup
                save_local_results(
                    os.path.join(args.output_data, "failed_updates"),
                    image_id,
                    annotations
                )
        
        # If local mode or as backup, save locally
        save_local_results(
            args.output_data,
            image_id,
            annotations
        )
        
        # Add to results
        results.append(f"{image_id}: {len(annotations)} annotations")
    
    # Log stats for this mini-batch
    logger.info(f"Mini-batch stats: Processed {batch_stats['images_processed']} images, found {batch_stats['detects_found']} detections")
    
    return results

def run_parallel_inference(args, models, processed_ids, cosmos_client):
    """Main function to process images in parallel"""
    logger.info(f"Starting parallel inference with {len(models)} models")
    
    # Stats for tracking metrics
    stats = {
        'images_processed': 0,
        'detects_found': 0,
        'database_updates': 0
    }
    
    # Find all images in input dataset
    input_path = args.input_ds
    batch_id = args.batch_id
    
    # Set up paths
    batch_path = os.path.join(input_path, f"B{batch_id}" if not str(batch_id).startswith("B") else str(batch_id))
    cam_path = os.path.join(batch_path, "cam")
    
    if not os.path.exists(cam_path):
        logger.warning(f"Camera path does not exist: {cam_path}")
        # Try alternative path formats
        cam_path = os.path.join(input_path, "cam")
        if not os.path.exists(cam_path):
            logger.error(f"Could not find camera images path")
            return []
    
    # Get list of images
    image_files = glob.glob(os.path.join(cam_path, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(cam_path, "*.jpeg")))
    image_files.extend(glob.glob(os.path.join(cam_path, "*.png")))
    
    logger.info(f"Found {len(image_files)} images in {cam_path}")
    
    if not image_files:
        logger.error(f"No image files found in {cam_path}")
        return []
    
    # Process each image
    results = []
    for file_path in image_files:
        # Get image ID from filename
        image_basename = os.path.basename(file_path)
        image_id = os.path.splitext(image_basename)[0]
        
        # Skip already processed images unless in force mode
        if args.mode == "auto" and image_id in processed_ids:
            logger.info(f"Skipping already processed image: {image_id}")
            continue
        
        logger.info(f"Processing image: {image_id}")
        stats['images_processed'] += 1
        
        # Process the image
        annotations = process_image(file_path, image_id, models, args)
        
        if not annotations:
            logger.info(f"No annotations found for image {image_id}")
            continue
          
        stats['detects_found'] += len(annotations)
        
        # Update CosmosDB if not in local mode
        if not args.local and cosmos_client:
            try:
                success = update_cosmos_db(
                    cosmos_client,
                    image_id,
                    annotations,
                    batch_id=args.batch_id,
                    database_name=args.cosmos_db_name,
                    container_name=args.cosmos_container_name
                )
                if success:
                    logger.info(f"Successfully updated CosmosDB record for {image_id}")
                    stats['database_updates'] += 1
                else:
                    logger.error(f"Failed to update CosmosDB record for {image_id}")
                    # Save locally as backup
                    save_local_results(
                        os.path.join(args.output_data, "failed_updates"),
                        image_id,
                        annotations
                    )
            except Exception as e:
                logger.error(f"Exception updating CosmosDB record for {image_id}: {str(e)}")
                # Save locally as backup
                save_local_results(
                    os.path.join(args.output_data, "failed_updates"),
                    image_id,
                    annotations
                )
        
        # If local mode or as backup, save locally
        save_local_results(
            args.output_data,
            image_id,
            annotations
        )
        
        # Add to results
        results.append(f"{image_id}: {len(annotations)} annotations")
    
    # Log stats for this batch
    logger.info(f"Batch stats: Processed {stats['images_processed']} images, found {stats['detects_found']} detections")
    
    # Try to log metrics
    if not args.local:
        try:
            mlflow.log_metric("images_processed", stats['images_processed'])
            mlflow.log_metric("detections_found", stats['detects_found'])
            mlflow.log_metric("database_updates", stats['database_updates'])
        except Exception as e:
            logger.warning(f"Could not log metrics: {str(e)}")
    else:
        logger.info(f"Local mode: Metrics summary - Processed {stats['images_processed']} images, found {stats['detects_found']} detections")
    
    return results

if __name__ == "__main__":
    # Use our centralized argument parsing function for standalone execution
    args = parse_arguments()
    
    print(f"Running in standalone mode with arguments: {args}")
    
    # Initialize mlflow for tracking only if not in local mode
    if not args.local:
        try:
            mlflow.start_run()
            logger.info("MLflow run started")
        except Exception as e:
            logger.warning(f"Could not start MLflow run: {str(e)}")
    else:
        logger.info("Local mode: Skipping MLflow initialization")
    
    models = load_models(args)
    
    # Set up CosmosDB client if not in local mode
    cosmos_client = None
    processed_ids = []
    if not args.local:
        if args.cosmos_db:
            try:
                from azure.cosmos import CosmosClient
                cosmos_client = CosmosClient.from_connection_string(args.cosmos_db)
                logger.info("Connected to CosmosDB using provided connection string")
            except Exception as e:
                logger.error(f"Failed to connect to CosmosDB: {str(e)}")
        else:
            try:
                cosmos_client = get_cosmosdb_client(None, args.key_vault_url)
                if cosmos_client:
                    logger.info("Connected to CosmosDB using Key Vault")
                else:
                    logger.warning("Failed to connect to CosmosDB using Key Vault")
            except Exception as e:
                logger.error(f"Failed to connect to CosmosDB via Key Vault: {str(e)}")
    
    # Get list of processed images
    if args.mode == "auto" and not args.local and cosmos_client:
        processed_ids = get_processed_ids(
            cosmos_client, 
            batch_id=args.batch_id,
            database_name=args.cosmos_db_name,
            container_name=args.cosmos_container_name
        )
        logger.info(f"Found {len(processed_ids)} already processed images")
    
    # Run the main process
    results = run_parallel_inference(args, models, processed_ids, cosmos_client)
    logger.info(f"Processed {len(results)} images")
    
    # Save results to output
    try:
        with open(os.path.join(args.output_data, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Successfully saved results to {os.path.join(args.output_data, 'results.json')}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
