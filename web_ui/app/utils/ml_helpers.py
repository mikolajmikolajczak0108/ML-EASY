"""
Utility functions for machine learning operations.
"""
import os
import torch
from fastai.vision.all import (
    load_learner, PILImage, ImageDataLoaders, vision_learner, 
    error_rate, Resize, RandomResizedCrop, imagenet_stats, models, 
    get_image_files, DataBlock, Categorize, CategoryBlock, ImageBlock
)
from flask import current_app


# Dictionary of available model architectures
AVAILABLE_ARCHITECTURES = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'mobilenet_v2': models.mobilenet_v2,
    'densenet121': models.densenet121
}

# Dictionary of dataset URLs
EXAMPLE_DATASETS = {
    'macbook_air_pro': 'https://storage.googleapis.com/ml-easy-datasets/macbooks_dataset.zip',
    'iphone_models': 'https://storage.googleapis.com/ml-easy-datasets/iphone_dataset.zip'
}


def get_available_models():
    """
    Get a list of available trained models.
    
    Returns:
        list: List of model names
    """
    # Get base model path from config
    base_models_path = current_app.config['MODEL_PATH']
    models = []
    
    print(f"DEBUG - Looking for models in: {base_models_path}")
    print(f"DEBUG - Directory exists: {os.path.exists(base_models_path)}")
    
    # Check the main models directory for .pkl and .h5 files
    if os.path.exists(base_models_path):
        print(f"DEBUG - Contents of {base_models_path}:")
        for item in os.listdir(base_models_path):
            item_path = os.path.join(base_models_path, item)
            print(f"DEBUG - Found: {item} ({os.path.isfile(item_path)=}, {os.path.isdir(item_path)=})")
            
            # Check if it's a direct model file (.pkl or .h5)
            if os.path.isfile(item_path) and (item.endswith('.pkl') or item.endswith('.h5')):
                # Add the filename without extension as the model name
                model_name = os.path.splitext(item)[0]
                if model_name not in models:
                    models.append(model_name)
                    print(f"DEBUG - Added file model: {model_name}")
                    
            # Check if it's a directory that might contain models
            elif os.path.isdir(item_path) and item != 'saved_models':
                print(f"DEBUG - Checking directory: {item}")
                # Check for model files within this directory
                for file in os.listdir(item_path):
                    print(f"DEBUG - Found file in {item}: {file}")
                    if file.endswith('.pkl') or file.endswith('.h5'):
                        # Use directory name as the model name
                        if item not in models:
                            models.append(item)
                            print(f"DEBUG - Added directory model: {item}")
                            break
    
    # Also check the saved_models subdirectory
    saved_models_path = os.path.join(base_models_path, 'saved_models')
    print(f"DEBUG - Checking saved_models at: {saved_models_path}")
    print(f"DEBUG - saved_models exists: {os.path.exists(saved_models_path)}")
    if os.path.exists(saved_models_path):
        print(f"DEBUG - Contents of {saved_models_path}:")
        for item in os.listdir(saved_models_path):
            item_path = os.path.join(saved_models_path, item)
            print(f"DEBUG - Found in saved_models: {item} ({os.path.isdir(item_path)=})")
            
            # Include directories in saved_models
            if os.path.isdir(item_path):
                # Check if the directory contains model files
                has_model_file = False
                for file in os.listdir(item_path):
                    print(f"DEBUG - Found file in {item}: {file}")
                    if file.endswith('.pkl') or file.endswith('.h5') or file == 'metadata.json':
                        has_model_file = True
                        print(f"DEBUG - Found valid model file: {file}")
                        break
                        
                if has_model_file and item not in models:
                    models.append(item)
                    print(f"DEBUG - Added saved_models model: {item}")
    
    print(f"DEBUG - Final models list: {models}")
    return models


def get_available_datasets():
    """
    Get a list of available datasets.
    
    Returns:
        dict: Dictionary with dataset names as keys and metadata as values
    """
    datasets_path = current_app.config['DATASET_PATH']
    datasets = {}
    
    for dataset_name in os.listdir(datasets_path):
        dataset_dir = os.path.join(datasets_path, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
            
        # Count classes and images
        classes = [d for d in os.listdir(dataset_dir) 
                  if os.path.isdir(os.path.join(dataset_dir, d))]
        total_images = sum(
            len(os.listdir(os.path.join(dataset_dir, cls))) 
            for cls in classes
        )
        
        datasets[dataset_name] = {
            'classes': classes,
            'num_classes': len(classes),
            'total_images': total_images
        }
        
    return datasets


def load_model(model_name):
    """
    Load a trained model.
    
    Args:
        model_name: Name of the model (directory or file without extension)
        
    Returns:
        model: Loaded model or None if loading fails
    """
    base_models_path = current_app.config['MODEL_PATH']
    
    try:
        # Check possible locations for this model
        
        # 1. Check if it's a .pkl file in the base models directory
        model_file = os.path.join(base_models_path, f"{model_name}.pkl")
        if os.path.exists(model_file) and os.path.isfile(model_file):
            return load_learner(model_file)
            
        # 2. Check if it's a .h5 file in the base models directory
        model_file = os.path.join(base_models_path, f"{model_name}.h5")
        if os.path.exists(model_file) and os.path.isfile(model_file):
            try:
                from tensorflow.keras.models import load_model as keras_load_model
                return keras_load_model(model_file)
            except ImportError:
                print("TensorFlow not installed - cannot load .h5 model")
                return None
                
        # 3. Check if it's a subdirectory of base_models_path
        model_dir = os.path.join(base_models_path, model_name)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            # Look for model files in the directory
            # Start with .pkl files
            for file in os.listdir(model_dir):
                if file.endswith('.pkl'):
                    return load_learner(os.path.join(model_dir, file))
                    
            # Then check for .h5 files
            for file in os.listdir(model_dir):
                if file.endswith('.h5'):
                    try:
                        from tensorflow.keras.models import load_model as keras_load_model
                        return keras_load_model(os.path.join(model_dir, file))
                    except ImportError:
                        print("TensorFlow not installed - cannot load .h5 model")
                        return None
        
        # 4. Check in the saved_models subdirectory
        saved_models_dir = os.path.join(base_models_path, 'saved_models', model_name)
        if os.path.exists(saved_models_dir) and os.path.isdir(saved_models_dir):
            # Try different file formats in order
            
            # First look for export.pkl (fastai format)
            model_file = os.path.join(saved_models_dir, 'export.pkl')
            if os.path.exists(model_file):
                return load_learner(model_file)
                
            # Then look for model.pkl (fastai format)
            model_file = os.path.join(saved_models_dir, 'model.pkl')
            if os.path.exists(model_file):
                return load_learner(model_file)
                
            # Then look for model.h5 (Keras/TensorFlow format)
            model_file = os.path.join(saved_models_dir, 'model.h5')
            if os.path.exists(model_file):
                try:
                    from tensorflow.keras.models import load_model as keras_load_model
                    return keras_load_model(model_file)
                except ImportError:
                    print("TensorFlow not installed - cannot load .h5 model")
                    return None
            
            # Check for any .h5 file
            h5_files = [f for f in os.listdir(saved_models_dir) if f.endswith('.h5')]
            if h5_files:
                try:
                    from tensorflow.keras.models import load_model as keras_load_model
                    return keras_load_model(os.path.join(saved_models_dir, h5_files[0]))
                except ImportError:
                    print("TensorFlow not installed - cannot load .h5 model")
                    return None
                    
            # Check for any .pkl file
            pkl_files = [f for f in os.listdir(saved_models_dir) if f.endswith('.pkl')]
            if pkl_files:
                return load_learner(os.path.join(saved_models_dir, pkl_files[0]))
                
        # Model not found in any location
        print(f"Model '{model_name}' not found in any location")
        return None
    except Exception as e:
        # Add logging for debugging
        print(f"Error loading model {model_name}: {str(e)}")
        return None


def create_datablock(dataset_path, img_size=224, batch_size=16):
    """
    Create a DataBlock for training.
    
    Args:
        dataset_path: Path to the dataset
        img_size: Size of images for training
        batch_size: Batch size for training
        
    Returns:
        tuple: (dls, dblock) DataLoaders and DataBlock
    """
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=lambda x: (x[:-2], x[-2:]),  # Simple validation split
        get_y=lambda x: x.parent.name,
        item_tfms=Resize(img_size),
        batch_tfms=RandomResizedCrop(img_size, min_scale=0.8)
    )
    
    dls = dblock.dataloaders(dataset_path, bs=batch_size)
    return dls, dblock 