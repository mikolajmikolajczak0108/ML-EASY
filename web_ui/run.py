#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to run the ML-EASY web application.
"""

import os
import sys
import subprocess
import importlib.util


def check_module(module_name):
    """Check if a module is installed."""
    return importlib.util.find_spec(module_name) is not None


def install_requirements():
    """Install required dependencies."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        return False


def check_dependencies():
    """Check required dependencies and install them if needed."""
    required_modules = ["flask", "torch", "fastai", "opencv-python", "numpy", "Pillow"]
    missing_modules = []
    
    for module in required_modules:
        module_name = "opencv-python" if module == "cv2" else module
        if not check_module(module):
            missing_modules.append(module_name)
    
    if missing_modules:
        print(f"Missing modules: {', '.join(missing_modules)}")
        print("Installing required dependencies...")
        if install_requirements():
            print("Dependencies installed successfully.")
        else:
            print("Error installing dependencies. Try installing them manually:")
            print("pip install -r requirements.txt")
            return False
    
    return True


def setup_folders():
    """Create required folders."""
    folders = ["uploads", "uploads/processed", "models", "datasets"]
    for folder in folders:
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder), exist_ok=True)


def run_app():
    """Run the Flask application."""
    try:
        print("Starting the web application...")
        from app import create_app
        app = create_app()
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting the application: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("=== ML-EASY - Machine Learning Made Easy ===")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create folders
    setup_folders()
    
    # Set environment variable for Flask
    os.environ['FLASK_ENV'] = 'development'
    
    # Run the application
    if not run_app():
        sys.exit(1) 