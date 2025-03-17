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
    
    logging.debug(f"Looking for models in: {base_models_path}")
    logging.debug(f"Directory exists: {os.path.exists(base_models_path)}")
    
    # Function to recursively find model files
    def find_model_files(directory):
        found_models = []
        if not os.path.exists(directory):
            logging.debug(f"Directory does not exist: {directory}")
            return found_models
            
        for root, dirs, files in os.walk(directory):
            logging.debug(f"Searching directory: {root}")
            logging.debug(f"Subdirectories found: {dirs}")
            logging.debug(f"Files found: {files}")
            
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
                        # Check if we're in a nested structure
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
                            model_name = os.path.basename(rel_path)
                    
                    # Clean up model name if needed
                    if model_name.lower() in ('model', 'models'):
                        # If model name is generic, use parent directory + filename
                        parent_dir = os.path.basename(os.path.dirname(rel_path))
                        if parent_dir and parent_dir.lower() not in ('model', 'models'):
                            model_name = f"{parent_dir}_{os.path.splitext(file)[0]}"
                        else:
                            model_name = os.path.splitext(file)[0]
                    
                    if model_name not in found_models:
                        found_models.append(model_name)
                        logging.debug(
                            f"Found model: {model_name} in {rel_path}/{file}"
                        )
                        
        return found_models
    
    # Find all model files recursively
    models = find_model_files(base_models_path)
    
    # If no models found, try looking for metadata.json files
    if not models:
        logging.debug(
            "No models found with standard extensions, looking for metadata"
        )
        for root, dirs, files in os.walk(base_models_path):
            for file in files:
                if file.lower() == 'metadata.json':
                    # Extract model name from directory containing metadata
                    rel_path = os.path.relpath(root, base_models_path)
                    model_dir = os.path.basename(rel_path)
                    if model_dir and model_dir not in models:
                        models.append(model_dir)
                        logging.debug(f"Found model via metadata: {model_dir}")
    
    logging.debug(f"Final models list: {models}")
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
        fastai.Learner: Loaded model
    """
    # Import here to avoid triggering Flask auto-reload
    # when PyTorch loads many internal modules
    import sys
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = None
    sys.stderr = None
    
    try:
        # Path to saved models
        models_dir = current_app.config['MODEL_PATH']
        saved_models_dir = os.path.join(models_dir, 'saved_models')
        model_dir = os.path.normpath(os.path.join(saved_models_dir, model_name))
        
        logging.info(f"Attempting to load model: {model_name} from path: {model_dir}")
        
        # Check if model exists as a directory
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            # Look for model.pkl first, then export.pkl as a fallback
            model_pkl_path = os.path.join(model_dir, 'model.pkl')
            export_path = os.path.join(model_dir, 'export.pkl')
            
            # Prioritize model.pkl
            if os.path.exists(model_pkl_path):
                model_path = model_pkl_path
                logging.info(f"Found model.pkl at: {model_path}")
            elif os.path.exists(export_path):
                model_path = export_path
                logging.info(f"Found export.pkl at: {model_path}")
            else:
                # Check for any .pkl file
                pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
                if pkl_files:
                    model_path = os.path.join(model_dir, pkl_files[0])
                    logging.info(f"Found PKL file at: {model_path}")
                else:
                    logging.error(f"No pickle model file found in directory: {model_dir}")
                    return None
        else:
            # Check for model files directly
            pkl_path = os.path.join(saved_models_dir, f"{model_name}.pkl")
            if os.path.exists(pkl_path):
                model_path = pkl_path
                logging.info(f"Found model at: {model_path}")
            else:
                logging.error(f"Model {model_name} not found in {saved_models_dir}")
                return None
        
        # Validate pickle file integrity
        try:
            # Check file size
            file_size = os.path.getsize(model_path)
            if file_size < 1000:  # Arbitrary small size that's too small for a model
                logging.error(f"Model file is too small ({file_size} bytes), likely corrupted")
                return None
                
            # Check pickle header - valid pickle files start with specific bytes
            with open(model_path, 'rb') as f:
                header = f.read(4)
                # Python 3 pickle protocol markers
                valid_headers = [b'\x80\x03', b'\x80\x04', b'\x80\x05']
                valid_pickle = any(header.startswith(h) for h in valid_headers)
                
                if not valid_pickle:
                    logging.error(f"Model file doesn't have a valid pickle header")
                    return None
        except Exception as e:
            logging.error(f"Error validating model file: {e}")
            return None
        
        # Load the model using fastai
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
                
                # Load the model quietly to avoid triggering Flask watchdog
                from fastai.learner import load_learner
                logging.info(f"Loading FastAI model from: {model_path}")
                
                try:
                    # Use a timeout to prevent hanging on corrupted files
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Model loading timed out")
                    
                    # Set timeout for model loading (5 seconds)
                    if sys.platform != 'win32':  # signal.SIGALRM not available on Windows
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(5)
                    
                    model = load_learner(model_path)
                    
                    # Cancel the alarm if loading succeeded
                    if sys.platform != 'win32':
                        signal.alarm(0)
                    
                    return model
                except TimeoutError:
                    logging.error("Model loading timed out - possible file corruption")
                    return None
                except EOFError:
                    logging.error("EOFError: Pickle file is truncated or corrupted")
                    return None
                
        except PermissionError as pe:
            logging.error(f"Permission denied when loading model: {pe}")
            return None
        except Exception as e:
            logging.error(f"Error loading fastai model: {e}")
            return None
    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr


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