"""
Utility functions for machine learning operations.
"""
import os
import logging
from flask import current_app
# We'll import specific modules when needed to avoid unnecessary dependencies

# Dictionary of available model architectures
AVAILABLE_ARCHITECTURES = {
    'resnet18': 'models.resnet18',
    'resnet34': 'models.resnet34',
    'resnet50': 'models.resnet50',
    'mobilenet_v2': 'models.mobilenet_v2',
    'densenet121': 'models.densenet121'
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
    Load a saved model by name
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        fastai.Learner or keras.Model: Loaded model
    """
    # Path to saved models
    models_dir = current_app.config['MODEL_PATH']
    saved_models_dir = os.path.join(models_dir, 'saved_models')
    model_dir = os.path.normpath(os.path.join(saved_models_dir, model_name))
    
    logging.info(f"Attempting to load model: {model_name} from path: {model_dir}")
    
    # Check if model exists
    if not os.path.exists(model_dir):
        # Check common file extensions directly
        h5_path = os.path.normpath(os.path.join(saved_models_dir, f"{model_name}.h5"))
        pkl_path = os.path.normpath(os.path.join(saved_models_dir, f"{model_name}.pkl"))
        export_path = os.path.normpath(os.path.join(saved_models_dir, f"{model_name}.export"))
        
        if os.path.exists(h5_path):
            model_path = h5_path
            logging.info(f"Found model at: {model_path}")
        elif os.path.exists(pkl_path):
            model_path = pkl_path
            logging.info(f"Found model at: {model_path}")
        elif os.path.exists(export_path):
            model_path = export_path
            logging.info(f"Found model at: {model_path}")
        else:
            logging.error(f"Model {model_name} not found in {saved_models_dir}")
            return None
    else:
        # Use the directory itself (for fastai models exported as directories)
        model_path = model_dir
        logging.info(f"Found model directory at: {model_path}")
    
    # Try to detect model type based on file extension or directory structure
    if model_path.endswith('.h5'):
        # Try multiple methods to detect TensorFlow
        tensorflow_found = False
        
        # Method 1: Direct import
        try:
            import tensorflow as tf
            tensorflow_found = True
            logging.info("TensorFlow detected via direct import")
        except ImportError:
            logging.warning("Failed to import TensorFlow directly")
            pass
        
        # Method 2: Check with subprocess if direct import fails
        if not tensorflow_found:
            try:
                import subprocess, sys
                result = subprocess.run(
                    [sys.executable, "-c", "import tensorflow"], 
                    capture_output=True, text=True, check=False
                )
                if result.returncode == 0:
                    tensorflow_found = True
                    logging.info("TensorFlow detected via subprocess check")
            except Exception as e:
                logging.warning(f"Failed to check TensorFlow via subprocess: {e}")
                pass
        
        # If we found TensorFlow, load the model
        if tensorflow_found:
            try:
                import tensorflow as tf
                logging.info(f"Loading Keras model from: {model_path}")
                model = tf.keras.models.load_model(model_path)
                return model
            except Exception as e:
                logging.error(f"Error loading Keras model: {e}")
                return None
        else:
            # TensorFlow not available
            error_msg = "TensorFlow required to load .h5 models. Please install TensorFlow first."
            logging.error(f"Error loading model {model_name}: {error_msg}")
            raise ImportError(error_msg)
    
    # Default to fastai for other model types
    try:
        # First try to check if we have read access
        try:
            if os.path.isdir(model_path):
                os.listdir(model_path)  # Check if we can read the directory
            else:
                with open(model_path, 'rb') as f:
                    pass  # Just check if we can open the file
        except PermissionError as pe:
            logging.error(f"Permission denied when accessing model path: {model_path}")
            raise PermissionError(f"Cannot access model file/directory: {pe}")
            
        # Temporarily suppress the pickle warning
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, 
                                   message="load_learner` uses Python's insecure pickle module")
            from fastai.learner import load_learner
            logging.info(f"Loading FastAI model from: {model_path}")
            model = load_learner(model_path)
            return model
    except PermissionError as pe:
        logging.error(f"Permission denied when loading model: {pe}")
        return None
    except Exception as e:
        logging.error(f"Error loading fastai model: {e}")
        
        # Try one last method - if it's a directory, look for an export.pkl file inside
        if os.path.isdir(model_path):
            try:
                export_path = os.path.join(model_path, 'export.pkl')
                if os.path.exists(export_path):
                    logging.info(f"Trying to load model from export.pkl: {export_path}")
                    # Temporarily suppress the pickle warning
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, 
                                            message="load_learner` uses Python's insecure pickle module")
                        model = load_learner(export_path)
                        return model
            except PermissionError as pe:
                logging.error(f"Permission denied when loading export.pkl: {pe}")
                return None
            except Exception as e2:
                logging.error(f"Error loading fastai model from export.pkl: {e2}")
        
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