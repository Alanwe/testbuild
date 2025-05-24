# Troubleshooting Guide for FacadeAI Inference Component

This document provides solutions for common issues encountered when setting up and running the FacadeAI Inference component.

## Table of Contents

1. [Environment Setup Issues](#environment-setup-issues)
2. [Model Loading Issues](#model-loading-issues)
3. [Runtime Errors](#runtime-errors)
4. [Image Processing Issues](#image-processing-issues)
5. [Azure ML and Cloud Integration Issues](#azure-ml-and-cloud-integration-issues)
6. [Performance Optimization](#performance-optimization)

## Environment Setup Issues

### Python Version Compatibility

**Issue**: Error message about incompatible Python version.

**Solution**:
- The Azure ML packages require Python 3.7-3.11 (depending on version)
- Check your Python version: `python --version`
- Create a new environment with the compatible version:
  ```bash
  conda create -n facade_env python=3.9
  conda activate facade_env
  ```

### Package Installation Failures

**Issue**: Dependency conflicts or installation errors.

**Solution**:
- Install packages in the following order to minimize conflicts:
  ```bash
  # Core packages first
  pip install numpy pandas pillow pyyaml
  
  # Then ML frameworks
  pip install torch torchvision
  
  # Then Azure packages
  pip install azureml-automl-dnn-vision==1.60.0
  
  # Then OpenCV
  pip install opencv-python-headless==4.8.1.78
  ```
- If a specific package fails, check its compatibility with your Python version

### Docker Build Issues

**Issue**: Docker build fails with errors.

**Solution**:
- Ensure Docker has sufficient resources allocated (memory and CPU)
- Check network connectivity for downloading base images and packages
- If specific package installation fails, try adding it to a separate RUN command:
  ```Dockerfile
  RUN pip install --no-cache-dir problematic-package
  ```

## Model Loading Issues

### Missing Module 'azureml'

**Issue**: Error message: `No module named 'azureml'`

**Solution**:
- Install the required Azure ML packages:
  ```bash
  pip install azureml-automl-dnn-vision==1.60.0
  ```
- For environments where installing azureml packages is not possible, create a dummy model as described in `src/old/create_dummy_model.py`

### Model File Structure Issues

**Issue**: Error loading model files or incorrect structure.

**Solution**:
- Check that the model directory contains all required files:
  ```
  models/Glazing-Defects/
    ├── MLmodel
    ├── conda.yaml
    ├── python_env.yaml  
    ├── python_model.pkl
    └── artifacts/
  ```
- Verify the MLmodel file has valid YAML syntax
- If using a custom model format, adapt the loading code in utils.py

### MLflow Version Mismatch

**Issue**: Warning about MLflow version incompatibility.

**Solution**:
- Try installing the specific MLflow version used to create the model:
  ```bash
  # Check MLmodel file for mlflow_version
  pip install mlflow==<version_from_model>
  ```
- If that's not possible, you may need to recreate or update the model

## Runtime Errors

### Out of Memory Errors

**Issue**: Process crashes with memory-related errors.

**Solution**:
- Reduce batch size or window size in `runlocal.sh`
- Process fewer images at once
- Increase memory allocation:
  - For Docker: Increase memory in Docker settings
  - For local runs: Close other memory-intensive applications

### GPU Related Issues

**Issue**: GPU not being utilized or CUDA errors.

**Solution**:
- Check if CUDA is available:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```
- Install CUDA-compatible version of PyTorch
- Ensure NVIDIA drivers are properly installed
- For Docker, use `--gpus all` flag when running the container

## Image Processing Issues

### Image Loading Failures

**Issue**: Errors when loading images: "Cannot read image" or similar.

**Solution**:
- Check image file permissions
- Verify image format is supported (JPG, PNG, etc.)
- Try using a different image loading method:
  ```python
  # Try with PIL instead of OpenCV
  from PIL import Image
  import numpy as np
  img = np.array(Image.open('image.jpg'))
  ```

### Empty or Unexpected Results

**Issue**: Processing completes but no annotations are generated.

**Solution**:
- Check confidence threshold - it may be too high:
  ```bash
  # Edit runlocal.sh and reduce the confidence threshold
  CONFIDENCE=30  # Lower from default of 50
  ```
- Verify the model works with the image format/resolution
- Try running on a known working test image

### Sliding Window Issues

**Issue**: Incorrect detection coordinates or missed detections at image boundaries.

**Solution**:
- Adjust sliding window parameters in `runlocal.sh`:
  ```bash
  WINDOW_SIZE=256  # Smaller window size
  OVERLAP=128      # Increased overlap
  ```
- Check the offset calculation in `utils.py` -> `recalculate_coordinates` function

## Azure ML and Cloud Integration Issues

### Authentication Failures

**Issue**: Unable to authenticate with Azure services.

**Solution**:
- Check that required environment variables are set:
  ```bash
  # For managed identity
  export AZURE_CLIENT_ID="your-client-id"
  export AZURE_TENANT_ID="your-tenant-id"
  export AZURE_CLIENT_SECRET="your-client-secret"
  ```
- Ensure the identity has appropriate permissions
- Try interactive login for testing:
  ```python
  from azure.identity import InteractiveBrowserCredential
  credential = InteractiveBrowserCredential()
  ```

### CosmosDB Connection Issues

**Issue**: Cannot connect to CosmosDB.

**Solution**:
- Verify connection string format
- Check network connectivity to Azure
- Ensure CosmosDB firewall allows your IP
- For local testing, use `--local true` to skip CosmosDB operations

### Pipeline Component Registration Issues

**Issue**: Component fails to register in Azure ML.

**Solution**:
- Check component.yml format
- Ensure all file paths in component.yml are relative to the directory
- Validate YAML syntax using a linter
- Check permissions in the Azure ML workspace

## Performance Optimization

### Slow Processing

**Issue**: Processing takes too long.

**Solution**:
- Optimize sliding window parameters:
  ```bash
  WINDOW_SIZE=768  # Larger window size
  OVERLAP=32       # Reduced overlap
  ```
- Enable GPU acceleration if available
- Process images in parallel by using multiple Docker containers
- Use a more powerful compute target in Azure ML

### High Resource Consumption

**Issue**: Process uses excessive CPU/memory.

**Solution**:
- Limit image resolution by downsampling large images
- Use OpenCV's optimized functions
- Free memory explicitly after processing each image
- Adjust batch size to process fewer images simultaneously

## Debugging Tips

### Enable Detailed Logging

Increase log verbosity for more information:

```bash
# Add to runlocal.sh before running component.py
export PYTHONUNBUFFERED=1  # Ensure logs are displayed immediately
```

### Run Components Separately

For troubleshooting, try running parts of the pipeline separately:

```bash
# Test model loading
python -c "import mlflow; model = mlflow.pyfunc.load_model('models/Glazing-Defects'); print('Model loaded successfully')"

# Test image loading
python -c "import cv2; img = cv2.imread('data/images/B3/cam/image.jpg'); print(f'Image loaded with shape {img.shape}')"
```

### Inspect Intermediate Results

Save intermediate processing steps for inspection:

```python
# Add to component.py in the process_image function
cv2.imwrite(f'debug/{image_id}_window_{i}.jpg', window)
```

### Use Visual Verification

Add a visualization step to verify detection results:

```bash
# Modify runlocal.sh to enable tracing
TRACE=true
```

This will save visualization images to the trace directory for inspection.