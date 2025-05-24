#!/usr/bin/env python3
"""
Environment verification script for FacadeAI development
This script checks if the environment is set up correctly and can load the required dependencies.
"""
import os
import sys
import importlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_import(module_name):
    """Try to import a module and return True if successful"""
    try:
        importlib.import_module(module_name)
        logger.info(f"✅ {module_name} imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import {module_name}: {str(e)}")
        return False

def check_model_structure(model_path):
    """Check if model structure exists and is valid"""
    if not os.path.exists(model_path):
        logger.error(f"❌ Model directory not found: {model_path}")
        return False
    
    required_files = ["MLmodel", "conda.yaml", "python_model.pkl"]
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            logger.error(f"❌ Required model file not found: {file_path}")
            return False
        else:
            logger.info(f"✅ Found model file: {file}")
    
    logger.info(f"✅ Model structure at {model_path} appears valid")
    return True

def check_directories():
    """Check if required directories exist"""
    dirs_to_check = ["data/images/B3/cam", "data/output", "models"]
    all_exist = True
    
    for dir_path in dirs_to_check:
        if not os.path.exists(dir_path):
            logger.error(f"❌ Required directory not found: {dir_path}")
            all_exist = False
        else:
            logger.info(f"✅ Directory exists: {dir_path}")
    
    return all_exist

if __name__ == "__main__":
    print("FacadeAI Development Environment Verification")
    print("-" * 50)
    
    # Check Python version
    py_version = sys.version.split()[0]
    if py_version.startswith("3.9"):
        logger.info(f"✅ Python version: {py_version}")
    else:
        logger.warning(f"⚠️ Python version is {py_version}, but 3.9.x is recommended")
    
    # Check critical imports
    critical_modules = [
        "mlflow", 
        "cv2", 
        "PIL", 
        "numpy", 
        "pandas", 
        "azure.cosmos"
    ]
    
    all_imports_ok = True
    for module in critical_modules:
        if not check_import(module):
            all_imports_ok = False
    
    # Check directories
    dirs_ok = check_directories()
    
    # Check model structure 
    model_path = "models/Dev-Model"
    if os.path.exists(model_path):
        model_ok = check_model_structure(model_path)
    else:
        logger.warning(f"⚠️ Development model not found at {model_path}")
        logger.info("You can create it using 'python src/old/create_dummy_model.py models/Dev-Model'")
        model_ok = False
    
    # Print summary
    print("\nVerification Summary:")
    print("-" * 50)
    if all_imports_ok and dirs_ok and model_ok:
        print("✅ Environment appears to be set up correctly!")
        print("You can now run './dev_runlocal.sh' to test the inference component.")
        sys.exit(0)
    else:
        print("⚠️ Some issues were detected with the environment setup.")
        print("Please check the log messages above and fix any issues.")
        if not all_imports_ok:
            print("- Missing dependencies: make sure to activate the conda environment")
        if not dirs_ok:
            print("- Missing directories: run setup_dev.sh to create them")
        if not model_ok:
            print("- Issue with model: create a development model or check its structure")
        sys.exit(1)