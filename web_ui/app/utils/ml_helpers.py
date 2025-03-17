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
    
    # Function to recursively find model files
    def find_model_files(directory):
        found_models = []
        if not os.path.exists(directory):
            print(f"DEBUG - Directory does not exist: {directory}")
            return found_models
            
        for root, dirs, files in os.walk(directory):
            print(f"DEBUG - Searching directory: {root}")
            print(f"DEBUG - Subdirectories found: {dirs}")
            print(f"DEBUG - Files found: {files}")
            
            # Check for model files in current directory
            for file in files:
                # Support various model file extensions
                if file.endswith(('.pkl', '.h5', '.pt', '.pth', '.model')):
                    # Extract model name from file path
                    rel_path = os.path.relpath(root, base_models_path)
                    
                    # Try different strategies to get a meaningful model name
                    if rel_path == '.':
                        # If in root directory, use filename without extension
                        model_name = os.path.splitext(file)[0]
                    else:
                        # Check if we're in a nested structure like saved_models/CatsDogs
                        path_parts = rel_path.split(os.sep)
                        if len(path_parts) > 1:
                            # Use the deepest directory name that's not 'saved_models'
                            for part in reversed(path_parts):
                                if part.lower() != 'saved_models':
                                    model_name = part
                                    break
                            else:
                                model_name = os.path.splitext(file)[0]
                        else:
                            # Use directory name
                            model_name = os.path.basename(root)
                    
                    # Clean up model name if needed
                    if model_name.lower() in ('model', 'models'):
                        # If model name is generic, use parent directory + filename
                        parent_dir = os.path.basename(os.path.dirname(root))
                        if parent_dir and parent_dir.lower() not in ('model', 'models'):
                            model_name = f"{parent_dir}_{os.path.splitext(file)[0]}"
                        else:
                            model_name = os.path.splitext(file)[0]
                    
                    if model_name not in found_models:
                        found_models.append(model_name)
                        print(f"DEBUG - Found model: {model_name} in {rel_path}/{file}")
                        
        return found_models
    
    # Find all model files recursively
    models = find_model_files(base_models_path)
    
    # If no models found, try looking for metadata.json files which might indicate models
    if not models:
        print("DEBUG - No models found with standard extensions, looking for metadata")
        for root, dirs, files in os.walk(base_models_path):
            for file in files:
                if file.lower() == 'metadata.json':
                    # Extract model name from directory containing metadata
                    model_dir = os.path.basename(root)
                    if model_dir and model_dir not in models:
                        models.append(model_dir)
                        print(f"DEBUG - Found model via metadata: {model_dir}")
    
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
        # First, check if TensorFlow is available for .h5 models
        has_tensorflow = False
        try:
            import tensorflow
            has_tensorflow = True
        except ImportError:
            print("Warning: TensorFlow not installed. Cannot load .h5 models.")
        
        # Check possible locations for this model
        
        # 1. Check if it's a .pkl file in the base models directory
        model_file = os.path.join(base_models_path, f"{model_name}.pkl")
        if os.path.exists(model_file) and os.path.isfile(model_file):
            return load_learner(model_file)
            
        # 2. Check if it's a .h5 file in the base models directory
        model_file = os.path.join(base_models_path, f"{model_name}.h5")
        if os.path.exists(model_file) and os.path.isfile(model_file):
            if has_tensorflow:
                from tensorflow.keras.models import load_model as keras_load_model
                return keras_load_model(model_file)
            else:
                raise ImportError("TensorFlow required to load .h5 models. Please install TensorFlow first.")
                
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
                    if has_tensorflow:
                        from tensorflow.keras.models import load_model as keras_load_model
                        return keras_load_model(os.path.join(model_dir, file))
                    else:
                        raise ImportError("TensorFlow required to load .h5 models. Please install TensorFlow first.")
        
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
                if has_tensorflow:
                    from tensorflow.keras.models import load_model as keras_load_model
                    return keras_load_model(model_file)
                else:
                    raise ImportError("TensorFlow required to load .h5 models. Please install TensorFlow first.")
            
            # Check for any .h5 file
            h5_files = [f for f in os.listdir(saved_models_dir) if f.endswith('.h5')]
            if h5_files:
                if has_tensorflow:
                    from tensorflow.keras.models import load_model as keras_load_model
                    return keras_load_model(os.path.join(saved_models_dir, h5_files[0]))
                else:
                    raise ImportError("TensorFlow required to load .h5 models. Please install TensorFlow first.")
                    
            # Check for any .pkl file
            pkl_files = [f for f in os.listdir(saved_models_dir) if f.endswith('.pkl')]
            if pkl_files:
                return load_learner(os.path.join(saved_models_dir, pkl_files[0]))
                
        # Model not found in any location
        raise FileNotFoundError(f"Model '{model_name}' not found in any location")
    except Exception as e:
        # Add proper error handling with contextual information
        import traceback
        print(f"Error loading model {model_name}: {str(e)}")
        print(traceback.format_exc())
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