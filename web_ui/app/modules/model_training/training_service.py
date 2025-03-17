import os
import time
import json
import logging
import traceback
import threading
import shutil
from pathlib import Path
import tempfile
import sys

from .utils import get_model_path, get_dataset_path

# Initialize logger with console output
logger = logging.getLogger(__name__)

# Add a console handler if not already present
if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)

# File lock for thread safety
_status_locks = {}

def get_status_lock(model_name):
    """Get a lock for a specific model's status file to prevent race conditions."""
    global _status_locks
    if model_name not in _status_locks:
        _status_locks[model_name] = threading.RLock()
    return _status_locks[model_name]

# Add file handler to logger for each training process
def setup_file_logger(model_name):
    """Set up a file logger for a specific training process."""
    try:
        # Get or create the logs directory
        logs_dir = os.path.join(get_model_path(), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create a log file for this model
        log_file = os.path.join(logs_dir, f"{model_name}.log")
        
        # Create a file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Create a formatter with timestamp
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Get logger for this model
        model_logger = logging.getLogger(f"model_training.{model_name}")
        model_logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicates
        for handler in model_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file:
                model_logger.removeHandler(handler)
        
        # Add the file handler to the logger
        model_logger.addHandler(file_handler)
        
        return model_logger
        
    except Exception as e:
        logger.error(f"Error setting up file logger for model {model_name}: {e}")
        return logging.getLogger(__name__)  # Return default logger as fallback


# Custom PrintLogger class that captures print output to both console and log file
class PrintLogger:
    def __init__(self, model_name):
        self.terminal = sys.stdout
        self.model_name = model_name
        self.logger = logging.getLogger(f"model_training.{model_name}")
        
    def write(self, message):
        self.terminal.write(message)
        if message.strip():  # Only log non-empty messages
            self.logger.info(message.strip())
            
    def flush(self):
        self.terminal.flush()


# Pre-import PyTorch and fastai libraries at module level
try:
    logger.info("Pre-importing libraries at module initialization...")
    
    import torch
    logger.info(f"PyTorch imported successfully: {torch.__version__}")
    
    # Import fastai vision library
    from fastai.vision.all import (
        ImageDataLoaders, 
        Resize, 
        vision_learner, 
        error_rate, 
        accuracy,
        RandomResizedCrop,
        aug_transforms
    )
    logger.info("fastai.vision.all imported successfully")
    
    # Import fastai metrics
    from fastai.metrics import Precision, Recall, F1Score
    logger.info("fastai.metrics imported successfully")
    
    # Import torchvision models
    import torchvision.models as models
    logger.info("torchvision.models imported successfully")
    
    # Flag to indicate successful imports
    LIBRARIES_IMPORTED = True
    logger.info("All ML libraries pre-imported successfully")
    
except ImportError as e:
    logger.error(f"Failed to pre-import ML libraries: {e}")
    logger.error(traceback.format_exc())
    LIBRARIES_IMPORTED = False
except Exception as e:
    logger.error(f"Unexpected error during library pre-import: {e}")
    logger.error(traceback.format_exc())
    LIBRARIES_IMPORTED = False


def train_model_task(model_name, dataset_name, architecture, epochs, batch_size, 
                    learning_rate, data_augmentation):
    """Background task to train a machine learning model with fastai/PyTorch."""
    # Set up logging for this training process
    model_logger = setup_file_logger(model_name)
    
    # Redirect print statements to also log to file
    original_stdout = sys.stdout
    sys.stdout = PrintLogger(model_name)
    
    try:
        # Print to console directly for visibility
        print(f"\n\n===== STARTING TRAINING: {model_name} =====")
        print(f"Dataset: {dataset_name}")
        print(f"Architecture: {architecture}")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}")
        print(f"Learning Rate: {learning_rate}, Data Augmentation: {data_augmentation}")
        print("=" * 50)
        
        # Disable PyTorch multiprocessing to avoid Windows errors
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
        
        # Disable multiprocessing for DataLoader (avoids Windows issues)
        os.environ['FASTAI_NUM_WORKERS'] = '0'
        
        # Extract dataset name if it's in dictionary format
        if isinstance(dataset_name, dict) and 'name' in dataset_name:
            dataset_name = dataset_name['name']
            logger.info(f"Extracted dataset name from dictionary: {dataset_name}")
            
        # First status update - checking libraries
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 5,
            'stage': 'checking_libraries',
            'message': 'Checking required libraries...'
        })
            
        logger.info(f"Started training model: {model_name} with dataset: {dataset_name}")
        
        # Check if libraries were successfully pre-imported
        if not LIBRARIES_IMPORTED:
            error_msg = "Required ML libraries were not properly imported during initialization. Check server logs."
            print(f"ERROR: {error_msg}")
            logger.error(error_msg)
            update_training_status(model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'import_error'
            })
            return False
            
        print("Using pre-imported libraries - ready to proceed with training")
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 10,
            'stage': 'libraries_ready',
            'message': 'ML libraries loaded successfully'
        })
        
        # 1. Prepare dataset
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 15,
            'stage': 'loading_dataset',
            'message': f'Loading dataset: {dataset_name}'
        })
        
        logger.info(f"Preparing dataset: {dataset_name}")
        print(f"\nPreparing dataset: {dataset_name}")
        dataset_path = os.path.join(get_dataset_path(), dataset_name)
        print(f"Dataset path: {dataset_path}")
        
        # Verify that dataset directory exists and contains image files
        if not os.path.exists(dataset_path):
            error_msg = f"Dataset directory not found: {dataset_path}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            update_training_status(model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'dataset_not_found'
            })
            return False
        else:
            print(f"Dataset directory exists: {dataset_path}")
            
        # Check if dataset has valid structure with class subdirectories
        subdirs = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
        
        print(f"Found {len(subdirs)} class directories: {subdirs}")
        
        if not subdirs:
            error_msg = f"No class subdirectories found in dataset: {dataset_path}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            update_training_status(model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'invalid_dataset_structure'
            })
            return False
            
        # Check if each class directory has images
        has_images = False
        image_counts = {}
        for subdir in subdirs:
            class_dir = os.path.join(dataset_path, subdir)
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            image_counts[subdir] = len(image_files)
            print(f"Class '{subdir}': {len(image_files)} images")
            if image_files:
                has_images = True
                
        if not has_images:
            error_msg = f"No image files found in dataset class directories: {dataset_path}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            update_training_status(model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'empty_dataset'
            })
            return False
        else:
            print(f"Dataset has images. Class counts: {image_counts}")
            
        # Set up augmentations based on user selection
        if data_augmentation:
            tfms = [
                RandomResizedCrop(224, min_scale=0.5),
                *aug_transforms()
            ]
            print(f"Using data augmentation with: RandomResizedCrop, aug_transforms()")
        else:
            tfms = [Resize(224)]
            print(f"Using minimal transformations: Resize(224)")
            
        # Create DataLoaders
        print("\n===== CREATING DATA LOADERS =====")
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 30,
            'stage': 'creating_dataloaders',
            'message': 'Creating DataLoaders...'
        })
        
        print(f"Creating DataLoaders with batch size: {batch_size}")
        try:
            print(f"Creating Path object for dataset: {dataset_path}")
            path = Path(dataset_path)
            
            print(f"Data transformations: {'Using augmentations' if data_augmentation else 'Minimal transforms'}")
            print(f"Batch size: {batch_size}, Workers: 0 (avoiding multiprocessing)")
            
            print("Calling ImageDataLoaders.from_folder()...")
            start_time = time.time()
            dls = ImageDataLoaders.from_folder(
                path,
                valid_pct=0.2,
                seed=42,
                bs=batch_size,
                item_tfms=Resize(224),
                batch_tfms=tfms,
                num_workers=0  # Avoid Windows multiprocessing issues
            )
            load_time = time.time() - start_time
            print(f"DataLoaders created in {load_time:.2f} seconds")
            
            print(f"DataLoaders created. Number of classes: {len(dls.vocab)}")
            print(f"Class names: {dls.vocab}")
            print(f"Total items - Train: {len(dls.train_ds)}, Validation: {len(dls.valid_ds)}")
            
            update_training_status(model_name, {
                'status': 'preparing',
                'progress': 40,
                'stage': 'dataloaders_created',
                'message': f'DataLoaders created with {len(dls.vocab)} classes'
            })
            
        except Exception as e:
            error_msg = f"Failed to create DataLoaders: {str(e)}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            print(traceback.format_exc())
            update_training_status(model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'dataloader_creation_error'
            })
            return False
            
        # Create the model
        print("\n===== CREATING MODEL =====")
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 50,
            'stage': 'creating_model',
            'message': f'Creating model with architecture: {architecture}'
        })
        
        try:
            print(f"Creating model with architecture: {architecture}")
            
            # Map architecture string to the correct architecture function
            arch_mapping = {
                'resnet18': models.resnet18,
                'resnet34': models.resnet34,
                'resnet50': models.resnet50,
                'mobile': models.mobilenet_v2,
                'MobileNetV2': models.mobilenet_v2,
                'ResNet50': models.resnet50,
                'EfficientNetB0': models.efficientnet_b0,
                'efficientnet': models.efficientnet_b0
            }
            
            if architecture not in arch_mapping:
                print(f"Warning: Unknown architecture '{architecture}', falling back to mobilenet_v2")
                architecture = 'mobile'
                
            arch_fn = arch_mapping.get(architecture, models.mobilenet_v2)
            print(f"Using model architecture function: {arch_fn.__name__}")
            
            # Create the learner
            print(f"Creating vision_learner with {len(dls.vocab)} classes")
            print("This might take a moment to download pre-trained weights...")
            
            start_time = time.time()
            try:
                # Add timeout protection to prevent indefinite hanging
                timeout_seconds = 120  # 2 minutes timeout
                
                # First check if we can create the model with pretrained=False to test basic functionality
                print(f"Testing architecture {arch_fn.__name__} compatibility...")
                test_model = None
                try:
                    # Set environment variable to cache models locally
                    os.environ['TORCH_HOME'] = os.path.join(get_model_path(), 'torch_cache')
                    print(f"TORCH_HOME set to: {os.environ['TORCH_HOME']}")
                    
                    # Ensure model cache directory exists
                    os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)
                    
                    # Network settings for better downloads
                    # Some networks block or throttle model downloads
                    try:
                        # Try to force a longer timeout for downloads
                        import urllib.request
                        urllib.request.socket.setdefaulttimeout(180)  # 3 minute timeout
                        print("Set default socket timeout to 180 seconds")
                        
                        # Check if we can access the PyTorch download server
                        test_url = "https://download.pytorch.org/models/hub/checkpoints/"
                        try:
                            print(f"Testing connection to {test_url}")
                            urllib.request.urlopen(test_url, timeout=10)
                            print("Connection to PyTorch download server successful")
                        except Exception as conn_err:
                            print(f"Warning: Cannot connect to PyTorch download server: {conn_err}")
                            print("Will attempt to use cached models if available")
                    except Exception as network_err:
                        print(f"Warning: Could not configure network settings: {network_err}")
                    
                    # Create architecture without pretrained weights
                    test_model = arch_fn(pretrained=False)
                    print(f"Basic architecture test passed for {arch_fn.__name__}")
                except Exception as arch_err:
                    print(f"Architecture test failed: {arch_err}")
                    print(f"Detailed error: {str(arch_err)}")
                    print(traceback.format_exc())
                    raise ValueError(f"Architecture {arch_fn.__name__} is not compatible: {str(arch_err)}")
                
                # Now try to create the full vision_learner
                print(f"Creating vision_learner with architecture: {arch_fn.__name__}")
                
                # Set a more restrictive set of metrics to avoid issues
                base_metrics = [accuracy, error_rate]
                
                try:
                    # First try with pretrained weights
                    print("Attempting to create model with pretrained weights...")
                    learn = vision_learner(
                        dls, 
                        arch_fn, 
                        metrics=base_metrics,
                        pretrained=True
                    )
                    print("Successfully created model with pretrained weights")
                except Exception as pretrained_err:
                    print(f"Error with pretrained weights: {pretrained_err}")
                    print("Trying without pretrained weights as fallback...")
                    
                    # Try without pretrained weights
                    learn = vision_learner(
                        dls, 
                        arch_fn, 
                        metrics=base_metrics,
                        pretrained=False
                    )
                    print("Successfully created model without pretrained weights")
                
                # Check if we've exceeded the timeout
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(f"Model creation timed out after {timeout_seconds} seconds")
                
                creation_time = time.time() - start_time
                print(f"Model created successfully in {creation_time:.2f} seconds")
                print(f"Base metrics: accuracy, error_rate")
                
                # Add additional metrics now that the model is created
                try:
                    learn.metrics.extend([Precision(), Recall(), F1Score()])
                    print("Extended metrics added: Precision, Recall, F1Score")
                except Exception as metrics_err:
                    print(f"Warning: Could not add extended metrics: {metrics_err}")
                
            except Exception as model_err:
                error_msg = str(model_err)
                print(f"Error creating vision_learner with {arch_fn.__name__}: {error_msg}")
                print(f"Detailed error: {traceback.format_exc()}")
                
                print("Trying with simpler ResNet18 architecture...")
                
                # Fallback to resnet18 which usually works
                fallback_start = time.time()
                try:
                    # Set a shorter timeout for fallback
                    fallback_timeout = 60  # 1 minute timeout
                    
                    learn = vision_learner(
                        dls, 
                        models.resnet18, 
                        metrics=[accuracy, error_rate],
                        pretrained=True
                    )
                    
                    # Check if we've exceeded the timeout
                    if time.time() - fallback_start > fallback_timeout:
                        raise TimeoutError(f"Fallback model creation timed out after {fallback_timeout} seconds")
                    
                    fallback_time = time.time() - fallback_start
                    print(f"Fallback model created successfully in {fallback_time:.2f} seconds")
                    print("Using ResNet18 with basic metrics (accuracy, error_rate)")
                    
                except Exception as fallback_err:
                    print(f"CRITICAL ERROR: Fallback model also failed: {fallback_err}")
                    print(f"Detailed error: {traceback.format_exc()}")
                    
                    # Ultimate fallback - try with no pretrained weights at all
                    print("Attempting last resort: ResNet18 without pretrained weights")
                    try:
                        learn = vision_learner(
                            dls, 
                            models.resnet18, 
                            metrics=[accuracy],
                            pretrained=False
                        )
                        print("Created ResNet18 without pretrained weights successfully")
                    except Exception as last_resort_err:
                        print(f"All fallback attempts failed: {last_resort_err}")
                        update_training_status(model_name, {
                            'status': 'error',
                            'error': f"Failed to create model: {error_msg}. Fallback also failed: {str(fallback_err)}",
                            'error_type': 'model_creation_error'
                        })
                        return False
            
            # Set learning rate
            print(f"Setting learning rate to {learning_rate}")
            learn.lr = learning_rate
            
            update_training_status(model_name, {
                'status': 'preparing',
                'progress': 60,
                'stage': 'model_created',
                'message': f'Model created with architecture: {architecture}'
            })
            
        except Exception as e:
            error_msg = f"Failed to create model: {str(e)}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            print(traceback.format_exc())
            update_training_status(model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'model_creation_error'
            })
            return False
        
        # 3. Training loop with progress updates
        update_training_status(model_name, {
            'status': 'training',
            'progress': 65,
            'stage': 'starting_training',
            'message': f'Starting training for {epochs} epochs with batch size {batch_size}'
        })
        
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        print(f"\n===== STARTING TRAINING =====")
        print(f"Model: {architecture}")
        print(f"Dataset: {dataset_name} ({len(dls.vocab)} classes)")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        # Custom callback to update training status
        class StatusCallback():
            """Custom callback to update training status."""
            
            def __init__(self, model_name):
                """Initialize the callback with model information."""
                self.model_name = model_name
                self.name = 'StatusCallback'  # Required by fastai
                self.current_epoch = 0
                self.total_epochs = 0
                self.training_start_time = None
                self.epoch_start_time = None
                self.learn = None
            
            def __call__(self, event_name):
                """Handle callback events."""
                try:
                    # Add more debug information
                    print(f"Callback event: {event_name}")
                    
                    # Delegate to specific event handlers
                    if event_name == 'before_fit':
                        self.before_fit()
                    elif event_name == 'before_epoch':
                        self.before_epoch()
                    elif event_name == 'after_epoch':
                        self.after_epoch()
                    elif event_name == 'after_fit':
                        self.after_fit()
                except Exception as e:
                    # Catch and log any errors in the callback
                    logger.error(f"Error in StatusCallback event {event_name}: {e}")
                    print(f"Error in callback ({event_name}): {e}")
            
            def before_fit(self):
                self.training_start_time = time.time()
                try:
                    print("\n>> Training starting - before_fit callback")
                    # Set total epochs based on the learner's n_epoch attribute if available
                    try:
                        if hasattr(self.learn, 'n_epoch'):
                            self.total_epochs = self.learn.n_epoch
                            print(f"Total epochs detected: {self.total_epochs}")
                    except Exception as e:
                        print(f"Could not determine total epochs: {e}")
                        
                    update_training_status(self.model_name, {
                        'status': 'training',
                        'progress': 65,
                        'stage': 'starting_training',
                        'epoch': 1,
                        'total_epochs': self.total_epochs,
                        'message': 'Starting model training...',
                        'metrics': {},
                        'training_start': time.time()
                    })
                    print("Training status updated: starting_training")
                except Exception as e:
                    logger.error(f"Error updating status in before_fit: {e}")
                    print(f"Error in before_fit: {e}")
            
            def before_epoch(self):
                try:
                    # Be defensive about accessing learn.epoch
                    epoch = 0
                    try:
                        if hasattr(self.learn, 'epoch'):
                            epoch = self.learn.epoch
                        self.current_epoch = epoch
                        self.epoch_start_time = time.time()
                    except Exception as e:
                        print(f"Error getting epoch information: {e}")
                        pass
                    
                    print(f"\n>> Starting epoch {epoch+1}/{self.total_epochs}")
                    # Calculate progress
                    base_progress = 65
                    freeze_progress = 5 
                    remaining_progress = 30
                    
                    if self.current_epoch == 0:
                        progress = base_progress + 2.5
                    else:
                        fine_tune_progress = self.current_epoch * (remaining_progress / max(1, self.total_epochs))
                        progress = base_progress + freeze_progress + fine_tune_progress
                    
                    phase = 'freeze' if self.current_epoch == 0 else 'fine_tune'
                    print(f"Training phase: {phase}, Progress: {progress:.1f}%")
                    print(f"Starting batch processing...")
                    
                    update_training_status(self.model_name, {
                        'status': 'training',
                        'progress': min(95, progress),
                        'stage': 'training_epoch',
                        'phase': phase,
                        'epoch': self.current_epoch + 1,
                        'total_epochs': self.total_epochs,
                        'message': f'Training epoch {self.current_epoch + 1}/{self.total_epochs}',
                        'metrics': {}
                    })
                except Exception as e:
                    logger.error(f"Error updating status in before_epoch: {e}")
                    print(f"Error in before_epoch: {e}")
            
            def after_epoch(self):
                try:
                    # Safely get metrics
                    epoch = self.current_epoch
                    train_loss = 0
                    valid_loss = 0
                    accuracy_val = 0
                    error_rate_val = 0
                    
                    # Calculate epoch duration
                    epoch_duration = 0
                    if self.epoch_start_time:
                        epoch_duration = time.time() - self.epoch_start_time
                        
                    print(f"\n>> Completed epoch {epoch+1}/{self.total_epochs} in {epoch_duration:.2f} seconds")
                    
                    # Safely extract metrics
                    try:
                        if hasattr(self.learn, 'recorder') and hasattr(self.learn.recorder, 'losses') and len(self.learn.recorder.losses) > 0:
                            train_loss = float(self.learn.recorder.losses[-1])
                            
                        if hasattr(self.learn, 'recorder') and hasattr(self.learn.recorder, 'values') and len(self.learn.recorder.values) > 0:
                            values = self.learn.recorder.values[-1]
                            if len(values) > 0:
                                valid_loss = float(values[0])
                            if len(values) > 1:
                                accuracy_val = float(values[1])
                            if len(values) > 2:
                                error_rate_val = float(values[2])
                                
                        print(f"Metrics - Train loss: {train_loss:.4f}, Validation loss: {valid_loss:.4f}")
                        print(f"Accuracy: {accuracy_val:.4f} ({accuracy_val*100:.2f}%), Error rate: {error_rate_val:.4f}")
                        print(f"Learning rate: {self.learn.opt.hypers[-1]['lr'] if hasattr(self.learn, 'opt') and hasattr(self.learn.opt, 'hypers') else 'unknown'}")
                        
                    except Exception as metric_err:
                        logger.error(f"Error extracting metrics: {metric_err}")
                        print(f"Error extracting metrics: {metric_err}")
                    
                    # Calculate progress percentage based on completed epochs
                    base_progress = 65
                    freeze_progress = 5
                    remaining_progress = 30
                    
                    if epoch == 0:
                        progress = base_progress + freeze_progress
                    else:
                        fine_tune_progress = (epoch + 1) * (remaining_progress / max(1, self.total_epochs))
                        progress = base_progress + freeze_progress + fine_tune_progress
                    
                    print(f"Training progress: {min(95, progress):.1f}%")
                    
                    # Calculate total training time so far
                    total_time = 0
                    if self.training_start_time:
                        total_time = time.time() - self.training_start_time
                        time_per_epoch = total_time / (epoch + 1)
                        estimated_remaining = time_per_epoch * (self.total_epochs - epoch - 1)
                        
                        print(f"Total training time: {format_duration(total_time)}")
                        print(f"Estimated time remaining: {format_duration(estimated_remaining)}")
                    
                    # Update training status with detailed metrics
                    update_training_status(self.model_name, {
                        'status': 'training',
                        'progress': min(95, progress),
                        'stage': 'epoch_complete',
                        'phase': 'freeze' if epoch == 0 else 'fine_tune',
                        'epoch': epoch + 1,
                        'total_epochs': self.total_epochs,
                        'message': f'Completed epoch {epoch + 1}/{self.total_epochs}',
                        'metrics': {
                            'loss': train_loss,
                            'accuracy': accuracy_val,
                            'val_loss': valid_loss,
                            'val_accuracy': 1 - error_rate_val,
                            'epoch': epoch + 1,
                            'total_epochs': self.total_epochs,
                            'epoch_duration': epoch_duration,
                            'total_duration': total_time
                        }
                    })
                    logger.info(f"Epoch {epoch+1}/{self.total_epochs} complete. Loss: {train_loss:.4f}, Accuracy: {accuracy_val:.4f}")
                except Exception as e:
                    logger.error(f"Error updating status in after_epoch: {e}")
                    print(f"Error in after_epoch: {e}")
            
            def after_fit(self):
                try:
                    # Update status when training is complete
                    total_time = 0
                    if self.training_start_time:
                        total_time = time.time() - self.training_start_time
                        
                    print("\n>> Training complete - after_fit callback")
                    print(f"Total training time: {format_duration(total_time)}")
                    
                    update_training_status(self.model_name, {
                        'status': 'saving',
                        'progress': 95,
                        'stage': 'training_complete',
                        'message': 'Training complete, saving model...',
                        'total_duration': total_time
                    })
                except Exception as e:
                    logger.error(f"Error updating status in after_fit: {e}")
                    print(f"Error in after_fit: {e}")
        
        # Register our custom callback
        status_cb = StatusCallback(model_name)
        learn.add_cb(status_cb)
        
        # Train the model with additional error handling
        try:
            # Use one-cycle policy for better stability
            learn.fine_tune(epochs, freeze_epochs=1)
            
            # Get final metrics from the recorder
            final_train_loss = 0
            final_valid_loss = 0
            final_accuracy = 0
            final_error_rate = 0
            
            # Safely try to extract metrics
            try:
                if hasattr(learn, 'recorder') and len(learn.recorder.losses) > 0:
                    final_train_loss = float(learn.recorder.losses[-1])
                
                if hasattr(learn, 'recorder') and len(learn.recorder.values) > 0:
                    if len(learn.recorder.values[-1]) > 0:
                        final_valid_loss = float(learn.recorder.values[-1][0])
                    if len(learn.recorder.values[-1]) > 1:
                        final_accuracy = float(learn.recorder.values[-1][1])
                    if len(learn.recorder.values[-1]) > 2:
                        final_error_rate = float(learn.recorder.values[-1][2])
            except Exception as metric_err:
                logger.error(f"Error extracting final metrics: {metric_err}")
                
        except Exception as train_err:
            logger.error(f"Error during training: {train_err}")
            logger.error(traceback.format_exc())
            update_training_status(model_name, {
                'status': 'error',
                'error': f"Training failed: {str(train_err)}",
                'error_type': 'training_error'
            })
            return False
        
        # 4. Save model
        try:
            update_training_status(model_name, {
                'status': 'saving',
                'progress': 95,
                'stage': 'saving_model',
                'message': f'Saving trained model: {model_name}',
                'metrics': {
                    'loss': final_train_loss,
                    'accuracy': final_accuracy,
                    'val_loss': final_valid_loss,
                    'val_accuracy': 1 - final_error_rate
                }
            })
            
            logger.info(f"Saving trained model: {model_name}")
            
            # Create model directory
            model_dir = os.path.join(get_model_path(), 'saved_models', model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model metadata
            metadata = {
                'model_name': model_name,
                'dataset': dataset_name,
                'architecture': architecture,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'data_augmentation': data_augmentation,
                'date_created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_classes': len(dls.vocab),
                'class_names': dls.vocab,
                'final_metrics': {
                    'loss': final_train_loss,
                    'accuracy': final_accuracy,
                    'val_loss': final_valid_loss,
                    'val_accuracy': 1 - final_error_rate
                }
            }
            
            update_training_status(model_name, {
                'status': 'saving',
                'progress': 97,
                'stage': 'saving_metadata',
                'message': 'Saving model metadata...'
            })
            
            with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Save fastai model with additional error handling
            try:
                update_training_status(model_name, {
                    'status': 'saving',
                    'progress': 98,
                    'stage': 'exporting_model',
                    'message': 'Exporting model to PKL format...'
                })
                
                model_pkl_path = os.path.join(model_dir, 'model.pkl')
                learn.export(model_pkl_path)
                logger.info(f"Saved fastai model to: {model_pkl_path}")
                
                # Also export to export.pkl for maximum compatibility
                export_pkl_path = os.path.join(model_dir, 'export.pkl')
                learn.export(export_pkl_path)
                logger.info(f"Saved fastai export model to: {export_pkl_path}")
                
            except Exception as export_err:
                logger.error(f"Error exporting model: {export_err}")
                # Continue with completion even if export fails
                
            # 5. Mark training as complete
            update_training_status(model_name, {
                'status': 'completed',
                'progress': 100,
                'stage': 'completed',
                'message': 'Training completed successfully!',
                'metrics': {
                    'loss': final_train_loss,
                    'accuracy': final_accuracy,
                    'val_loss': final_valid_loss,
                    'val_accuracy': 1 - final_error_rate
                },
                'model_info': {
                    'model_name': model_name,
                    'dataset': dataset_name,
                    'architecture': architecture,
                    'num_classes': len(dls.vocab),
                    'class_names': dls.vocab,
                    'training_time': int(time.time() - float(status_cb.training_start_time)) if hasattr(status_cb, 'training_start_time') and status_cb.training_start_time else None
                }
            })
            
            logger.info(f"Model training completed successfully: {model_name}")
            return True
            
        except Exception as save_err:
            logger.error(f"Error saving model: {save_err}")
            update_training_status(model_name, {
                'status': 'error',
                'error': f"Failed to save model: {str(save_err)}",
                'error_type': 'save_error'
            })
            return False
        
    except Exception as e:
        logger.error(f"Error training model {model_name}: {e}")
        logger.error(traceback.format_exc())
        
        # Log to model-specific log file as well
        model_logger.error(f"Error training model {model_name}: {e}")
        model_logger.error(traceback.format_exc())
        
        # Update status to error
        update_training_status(model_name, {
            'status': 'error',
            'error': str(e)
        })
        
        return False
    finally:
        # Restore original stdout
        sys.stdout = original_stdout


def update_training_status(model_name, status_update):
    """Update the training status for a model with thread safety."""
    # Get lock for this model's status file
    lock = get_status_lock(model_name)
    
    with lock:  # Ensure thread-safe access
        try:
            # Get path to status file
            status_dir = os.path.join(get_model_path(), 'training_status')
            os.makedirs(status_dir, exist_ok=True)
            status_file = os.path.join(status_dir, f"{model_name}.json")
            
            # Start with a clean base status
            status = {
                'model_name': model_name,
                'started_at': str(time.time()),
                'status': 'initializing',
                'progress': 0
            }
            
            # Try to read existing status if available
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        file_content = f.read().strip()
                        if file_content:  # Make sure file isn't empty
                            existing = json.loads(file_content)
                            if isinstance(existing, dict):
                                status.update(existing)
                except (json.JSONDecodeError, ValueError) as e:
                    # File exists but is corrupt - log and recreate
                    logger.error(f"Error reading status file {status_file}, creating new status: {e}")
                    # Backup the corrupt file for debugging
                    if os.path.exists(status_file) and os.path.getsize(status_file) > 0:
                        corrupt_file = f"{status_file}.corrupt.{int(time.time())}"
                        try:
                            shutil.copy2(status_file, corrupt_file)
                            logger.info(f"Backed up corrupt status file to {corrupt_file}")
                        except Exception as backup_err:
                            logger.error(f"Failed to backup corrupt status file: {backup_err}")
            
            # Update status with new values
            status.update(status_update)
            
            # Convert to JSON-serializable format
            safe_status = safe_json_status(status)
            
            # If training completed, add completion time
            if status_update.get('status') == 'completed':
                safe_status['completed_at'] = str(time.time())
            
            # Write to a temporary file first to ensure atomic update
            try:
                # Create a temp file in the same directory for atomic move
                fd, temp_path = tempfile.mkstemp(dir=status_dir, prefix=f"{model_name}_", suffix=".tmp")
                with os.fdopen(fd, 'w') as temp_file:
                    json.dump(safe_status, temp_file, indent=2, default=str)
                
                # Make sure the temp file was written successfully
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    # Use atomic rename on the temp file
                    if os.name == 'nt':  # Windows
                        # Windows needs special handling for atomic replace
                        if os.path.exists(status_file):
                            os.replace(temp_path, status_file)
                        else:
                            os.rename(temp_path, status_file)
                    else:
                        # Unix systems support atomic replace
                        os.rename(temp_path, status_file)
                else:
                    logger.error(f"Temporary file {temp_path} was not created properly")
                    # Fallback to direct write if temp file approach failed
                    with open(status_file, 'w') as f:
                        json.dump(safe_status, f, indent=2, default=str)
            except Exception as temp_err:
                logger.error(f"Error with temp file approach: {temp_err}")
                # Fallback to direct write
                with open(status_file, 'w') as f:
                    json.dump(safe_status, f, indent=2, default=str)
                
            logger.debug(f"Updated training status for {model_name}: {status_update.get('status')}")
        except Exception as e:
            logger.error(f"Error updating training status: {e}")
            logger.error(traceback.format_exc())


def get_training_status(model_name):
    """Get the current training status for a model."""
    # Get lock for this model's status file
    lock = get_status_lock(model_name)
    
    with lock:  # Ensure thread-safe access
        try:
            status_file = os.path.join(get_model_path(), 'training_status', f"{model_name}.json")
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            return json.loads(content)
                        else:
                            return {'status': 'empty_file'}
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in status file: {e}")
                    return {'status': 'error', 'error': f"Corrupted status file: {str(e)}"}
            return {'status': 'not_found'}
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return {'status': 'error', 'error': str(e)}


def start_model_training(model_name, dataset_name, architecture, epochs, batch_size, 
                        learning_rate, data_augmentation):
    """Start training a model in a background thread."""
    try:
        # Extract dataset name if it's in dictionary format
        if isinstance(dataset_name, dict) and 'name' in dataset_name:
            dataset_name = dataset_name['name']
            logger.info(f"Extracted dataset name from dictionary: {dataset_name}")
            
        # Create initial status
        update_training_status(model_name, {
            'status': 'initializing',
            'progress': 0,
            'model_name': model_name,
            'dataset': dataset_name,
            'architecture': architecture,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Start training in a background thread
        training_thread = threading.Thread(
            target=train_model_task,
            args=(model_name, dataset_name, architecture, epochs, 
                  batch_size, learning_rate, data_augmentation)
        )
        training_thread.daemon = True
        training_thread.start()
        
        logger.info(f"Started training thread for model: {model_name}")
        return True
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        logger.error(traceback.format_exc())
        return False


def get_models():
    """Get a list of all trained models."""
    try:
        # Get base model path
        base_models_path = get_model_path()
        models = []
        
        # Check the main models directory for model directories or files
        if os.path.exists(base_models_path):
            for item in os.listdir(base_models_path):
                item_path = os.path.join(base_models_path, item)
                
                # Include directories that are not saved_models
                if os.path.isdir(item_path) and item != 'saved_models' and item != 'training_status':
                    # Check if the directory contains model files
                    for file in os.listdir(item_path):
                        if file.endswith('.pkl'):
                            if item not in models:
                                models.append(item)
                                break
        
        # Also check the saved_models subdirectory
        saved_models_path = os.path.join(base_models_path, 'saved_models')
        if os.path.exists(saved_models_path):
            for item in os.listdir(saved_models_path):
                item_path = os.path.join(saved_models_path, item)
                
                # Include directories in saved_models
                if os.path.isdir(item_path):
                    # Check if the directory contains model files
                    has_model_file = False
                    
                    for file in os.listdir(item_path):
                        if file.endswith('.pkl') or file == 'export.pkl':
                            has_model_file = True
                            break
                        elif file == 'metadata.json':
                            has_model_file = True
                            
                    if has_model_file and item not in models:
                        models.append(item)
        
        return models
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        logger.error(traceback.format_exc())
        return []


def get_model_metadata(model_name):
    """Get metadata for a trained model."""
    try:
        metadata_file = os.path.join(get_model_path(), 'saved_models', 
                                    model_name, 'metadata.json')
        if not os.path.exists(metadata_file):
            return None
        
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error getting model metadata: {e}")
        logger.error(traceback.format_exc())
        return None


# Helper to safely convert class information to JSON-serializable format
def safe_json_status(status_dict):
    """Convert status dictionary to JSON-serializable format."""
    if not isinstance(status_dict, dict):
        return {'status': 'error', 'message': 'Invalid status data'}
        
    try:
        # Make a deep copy to avoid modifying the original
        import copy
        status = copy.deepcopy(status_dict)
        
        # Helper function to process nested objects
        def make_serializable(obj):
            if hasattr(obj, 'items') and not isinstance(obj, dict):
                # Convert dict-like objects (like CategoryMap) to list or dict
                try:
                    return list(obj)
                except:
                    try:
                        return dict(obj)
                    except:
                        return str(obj)
            elif isinstance(obj, dict):
                # Process nested dictionaries
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                # Process lists and tuples
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, float):
                # Handle NaN, Infinity which are not JSON serializable
                if not isinstance(obj, bool) and not (float('-inf') < obj < float('inf')):
                    return str(obj)
                return obj
            elif hasattr(obj, '__dict__'):
                # Handle class instances
                return str(obj)
            else:
                # Return basic types as is
                return obj
                
        # Process the entire status dictionary
        serializable_status = make_serializable(status)
        
        # Validate by trying to serialize to JSON string and back
        try:
            json_str = json.dumps(serializable_status)
            json.loads(json_str)  # Make sure it can be parsed back
            return serializable_status
        except (TypeError, ValueError) as json_err:
            logger.error(f"JSON serialization validation failed: {json_err}")
            # If validation fails, return a simple safe version
            return {
                'status': status.get('status', 'error'),
                'message': status.get('message', 'Serialization error'),
                'model_name': status.get('model_name', ''),
                'progress': status.get('progress', 0)
            }
        
    except Exception as e:
        logger.error(f"Error preparing status for JSON: {e}")
        # Return a simplified version that will definitely serialize
        return {
            'status': status_dict.get('status', 'error'),
            'message': status_dict.get('message', 'Error preparing status'),
            'model_name': status_dict.get('model_name', ''),
            'progress': status_dict.get('progress', 0)
        } 

# Helper function to format durations 
def format_duration(seconds):
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours" 