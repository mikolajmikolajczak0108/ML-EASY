import os
import time
import json
import logging
import traceback
import threading
from pathlib import Path

from .utils import get_model_path, get_dataset_path

# Initialize logger
logger = logging.getLogger(__name__)


def train_model_task(model_name, dataset_name, architecture, epochs, batch_size, 
                    learning_rate, data_augmentation):
    """Background task to train a machine learning model with fastai/PyTorch."""
    try:
        # Disable PyTorch multiprocessing to avoid Windows errors
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
        
        # Disable multiprocessing for DataLoader (avoids Windows issues)
        # Must be set before importing torch
        os.environ['FASTAI_NUM_WORKERS'] = '0'
        
        # Extract dataset name if it's in dictionary format
        if isinstance(dataset_name, dict) and 'name' in dataset_name:
            dataset_name = dataset_name['name']
            logger.info(f"Extracted dataset name from dictionary: {dataset_name}")
            
        # First status update - starting model preparation
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 5,
            'stage': 'importing_libraries',
            'message': 'Importing required libraries...'
        })
            
        logger.info(f"Started training model: {model_name} with dataset: {dataset_name}")
        
        # Import fastai/torch here to avoid blocking the main thread during import
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, random_split
            import torchvision.transforms as transforms
            import torchvision.models as models
            from torchvision.datasets import ImageFolder
            from fastai.vision.all import (
                ImageDataLoaders, 
                Resize, 
                vision_learner, 
                error_rate, 
                accuracy,
                RandomResizedCrop
            )
            from fastai.metrics import Precision, Recall, F1Score
            
            # Libraries imported successfully
            update_training_status(model_name, {
                'status': 'preparing',
                'progress': 10,
                'stage': 'libraries_imported',
                'message': 'Libraries imported successfully'
            })
            
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            update_training_status(model_name, {
                'status': 'error',
                'error': f"Failed to import required libraries: {e}",
                'error_type': 'import_error'
            })
            return False
        
        # 1. Prepare dataset
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 15,
            'stage': 'loading_dataset',
            'message': f'Loading dataset: {dataset_name}'
        })
        
        logger.info(f"Preparing dataset: {dataset_name}")
        dataset_path = os.path.join(get_dataset_path(), dataset_name)
        
        # Verify that dataset directory exists and contains image files
        if not os.path.exists(dataset_path):
            error_msg = f"Dataset directory not found: {dataset_path}"
            logger.error(error_msg)
            update_training_status(model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'dataset_not_found'
            })
            return False
            
        # Check if dataset has valid structure with class subdirectories
        subdirs = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
        
        if not subdirs:
            error_msg = f"No class subdirectories found in dataset: {dataset_path}"
            logger.error(error_msg)
            update_training_status(model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'invalid_dataset_structure'
            })
            return False
            
        # Check if each class directory has images
        has_images = False
        for subdir in subdirs:
            class_dir = os.path.join(dataset_path, subdir)
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            if image_files:
                has_images = True
                break
                
        if not has_images:
            error_msg = f"No image files found in dataset class directories: {dataset_path}"
            logger.error(error_msg)
            update_training_status(model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'empty_dataset'
            })
            return False
            
        # Set up augmentations based on user selection
        if data_augmentation:
            tfms = [
                Resize(224),
                RandomResizedCrop(224, min_scale=0.8)
            ]
        else:
            tfms = [Resize(224)]
        
        # Create fastai DataLoaders
        try:
            # Path object is needed for fastai
            path = Path(dataset_path)
            
            update_training_status(model_name, {
                'status': 'preparing',
                'progress': 20,
                'stage': 'creating_dataloaders',
                'message': 'Creating data loaders from images'
            })
            
            # Create fastai DataLoaders
            dls = ImageDataLoaders.from_folder(
                path,
                valid_pct=0.2,                   # 20% validation split
                item_tfms=tfms,                  # Transformations applied to each item
                batch_size=batch_size,
                seed=42                          # For reproducibility
            )
            
            # Get class names
            class_names = dls.vocab
            num_classes = len(class_names)
            
            logger.info(f"Successfully loaded dataset with {num_classes} classes: {class_names}")
            logger.info(f"Training batches: {len(dls.train)}, Validation batches: {len(dls.valid)}")
            
            update_training_status(model_name, {
                'status': 'preparing',
                'progress': 30,
                'stage': 'dataset_ready',
                'message': f'Dataset ready with {num_classes} classes',
                'details': {
                    'classes': class_names,
                    'num_classes': num_classes,
                    'training_batches': len(dls.train),
                    'validation_batches': len(dls.valid)
                }
            })
            
        except Exception as e:
            logger.error(f"Error creating DataLoaders: {e}")
            logger.error(traceback.format_exc())
            update_training_status(model_name, {
                'status': 'error',
                'error': f"Error loading dataset: {str(e)}",
                'error_type': 'dataset_error'
            })
            raise
        
        # 2. Create model architecture
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 40,
            'stage': 'downloading_model',
            'message': f'Downloading pre-trained {architecture} model weights...'
        })
        
        logger.info(f"Building model architecture: {architecture}")
        
        # Map architecture string to fastai model
        try:
            # First update status to inform user
            update_training_status(model_name, {
                'status': 'preparing',
                'progress': 45,
                'stage': 'downloading_model',
                'message': f'Downloading pre-trained {architecture} model weights...'
            })
            
            # Use fastai's timm models when possible for better compatibility
            if architecture == "ResNet50":
                from fastai.vision.all import resnet50
                model_arch = resnet50
            elif architecture == "MobileNetV2":
                try:
                    # Use direct torchvision model for MobileNetV2
                    # First create the base model with pretrained weights
                    from torchvision import models
                    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
                    # Use the model directly (fastai will handle the head)
                    model_arch = model
                except Exception as mobile_err:
                    logger.error(f"Error with MobileNetV2: {mobile_err}")
                    # Fallback to ResNet18 which is very reliable
                    from fastai.vision.all import resnet18
                    logger.warning("MobileNetV2 failed, using ResNet18 instead")
                    model_arch = resnet18
                    architecture = "ResNet18 (fallback)"
            elif architecture == "EfficientNetB0":
                try:
                    from timm import create_model
                    model_arch = create_model('efficientnet_b0', pretrained=True)
                except ImportError:
                    # Fallback to a model we know works with fastai
                    from fastai.vision.all import resnet34
                    logger.warning("EfficientNetB0 not available, using ResNet34 instead")
                    model_arch = resnet34
                    architecture = "ResNet34" 
            elif architecture == "VGG16":
                from fastai.vision.all import vgg16
                model_arch = vgg16
            elif architecture == "InceptionV3":
                try:
                    # Special handling for Inception due to auxiliary outputs
                    model = models.inception_v3(weights='IMAGENET1K_V1', aux_logits=False)
                    model_arch = model
                except Exception:
                    # Fallback to a more compatible model
                    from fastai.vision.all import resnet34
                    logger.warning("InceptionV3 not available, using ResNet34 instead")
                    model_arch = resnet34
                    architecture = "ResNet34"
            elif architecture == "DenseNet121":
                from fastai.vision.all import densenet121
                model_arch = densenet121
            else:
                # Default to ResNet34 - most reliable with fastai
                from fastai.vision.all import resnet34
                logger.warning(f"Architecture {architecture} not recognized, using ResNet34")
                model_arch = resnet34
                architecture = "ResNet34"
            
            update_training_status(model_name, {
                'status': 'preparing',
                'progress': 50,
                'stage': 'creating_model',
                'message': f'Creating {architecture} model for {num_classes} classes'
            })
            
            # Create fastai learner with proper error handling
            try:
                learn = vision_learner(dls, model_arch, metrics=[accuracy, error_rate])
                
                # Add additional metrics after successful creation
                learn.metrics.extend([Precision(), Recall(), F1Score()])
                
                # Set learning rate
                learn.lr = learning_rate
                
                update_training_status(model_name, {
                    'status': 'preparing',
                    'progress': 60,
                    'stage': 'model_ready',
                    'message': 'Model architecture ready for training'
                })
            except Exception as model_err:
                logger.error(f"Error creating vision_learner: {model_err}")
                # Try more basic approach for problematic architectures
                try:
                    # Fall back to a simpler, more reliable model
                    from fastai.vision.all import resnet18
                    logger.warning(f"Falling back to ResNet18 due to compatibility issues")
                    learn = vision_learner(dls, resnet18, metrics=[accuracy, error_rate])
                    learn.metrics.extend([Precision(), Recall(), F1Score()])
                    learn.lr = learning_rate
                    architecture = "ResNet18 (fallback)"
                    
                    update_training_status(model_name, {
                        'status': 'preparing',
                        'progress': 60,
                        'stage': 'model_ready',
                        'message': 'Using fallback model architecture due to compatibility issues'
                    })
                except Exception as fallback_err:
                    logger.error(f"Even fallback model failed: {fallback_err}")
                    raise RuntimeError(f"Cannot create model learner: {model_err}. Fallback also failed: {fallback_err}")
        
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            logger.error(traceback.format_exc())
            update_training_status(model_name, {
                'status': 'error',
                'error': f"Error creating model: {str(e)}",
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
        
        # Custom callback to update training status
        class StatusCallback():
            def __init__(self, model_name):
                self.model_name = model_name
                self.current_epoch = 0
                self.total_epochs = epochs
                self.training_start_time = None
                # Add name attribute for fastai compatibility
                self.name = 'status_callback'
                self.order = 0  # Lower order runs earlier
                
            def __call__(self, event_name):
                """Make the callback callable as required by fastai."""
                # Map event names to handler methods
                event_handlers = {
                    'before_fit': self.before_fit,
                    'before_epoch': self.before_epoch,
                    'after_epoch': self.after_epoch,
                    'after_fit': self.after_fit
                }
                
                # Call appropriate handler if it exists
                if event_name in event_handlers:
                    try:
                        event_handlers[event_name]()
                    except Exception as e:
                        logger.error(f"Error in callback {event_name}: {e}")
                
            def before_fit(self):
                self.training_start_time = time.time()
                try:
                    update_training_status(self.model_name, {
                        'status': 'training',
                        'progress': 65,
                        'stage': 'starting_training',
                        'epoch': 1,
                        'total_epochs': self.total_epochs,
                        'message': 'Starting model training...',
                        'metrics': {}
                    })
                except Exception as e:
                    logger.error(f"Error updating status in before_fit: {e}")
                
            def before_epoch(self):
                try:
                    # Be defensive about accessing learn.epoch
                    epoch = 0
                    try:
                        if hasattr(learn, 'epoch'):
                            epoch = learn.epoch
                        self.current_epoch = epoch
                    except:
                        pass
                        
                    # Calculate progress
                    base_progress = 65
                    freeze_progress = 5 
                    remaining_progress = 30
                    
                    if self.current_epoch == 0:
                        progress = base_progress + 2.5
                    else:
                        fine_tune_progress = self.current_epoch * (remaining_progress / max(1, self.total_epochs))
                        progress = base_progress + freeze_progress + fine_tune_progress
                    
                    update_training_status(self.model_name, {
                        'status': 'training',
                        'progress': min(95, progress),
                        'stage': 'training_epoch',
                        'phase': 'freeze' if self.current_epoch == 0 else 'fine_tune',
                        'epoch': self.current_epoch + 1,
                        'total_epochs': self.total_epochs,
                        'message': f'Training epoch {self.current_epoch + 1}/{self.total_epochs}',
                        'metrics': {}
                    })
                except Exception as e:
                    logger.error(f"Error updating status in before_epoch: {e}")
                
            def after_epoch(self):
                try:
                    # Safely get metrics
                    epoch = self.current_epoch
                    train_loss = 0
                    valid_loss = 0
                    accuracy_val = 0
                    error_rate_val = 0
                    
                    # Safely extract metrics
                    try:
                        if hasattr(learn, 'recorder') and hasattr(learn.recorder, 'losses') and len(learn.recorder.losses) > 0:
                            train_loss = float(learn.recorder.losses[-1])
                            
                        if hasattr(learn, 'recorder') and hasattr(learn.recorder, 'values') and len(learn.recorder.values) > 0:
                            values = learn.recorder.values[-1]
                            if len(values) > 0:
                                valid_loss = float(values[0])
                            if len(values) > 1:
                                accuracy_val = float(values[1])
                            if len(values) > 2:
                                error_rate_val = float(values[2])
                    except Exception as metric_err:
                        logger.error(f"Error extracting metrics: {metric_err}")
                    
                    # Calculate progress percentage based on completed epochs
                    base_progress = 65
                    freeze_progress = 5
                    remaining_progress = 30
                    
                    if epoch == 0:
                        progress = base_progress + freeze_progress
                    else:
                        fine_tune_progress = (epoch + 1) * (remaining_progress / max(1, self.total_epochs))
                        progress = base_progress + freeze_progress + fine_tune_progress
                    
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
                            'total_epochs': self.total_epochs
                        }
                    })
                    logger.info(f"Epoch {epoch+1}/{self.total_epochs} complete. Loss: {train_loss:.4f}, Accuracy: {accuracy_val:.4f}")
                except Exception as e:
                    logger.error(f"Error updating status in after_epoch: {e}")
            
            def after_fit(self):
                try:
                    # Update status when training is complete
                    update_training_status(self.model_name, {
                        'status': 'saving',
                        'progress': 95,
                        'stage': 'training_complete',
                        'message': 'Training complete, saving model...',
                    })
                except Exception as e:
                    logger.error(f"Error updating status in after_fit: {e}")
        
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
                'num_classes': num_classes,
                'class_names': class_names,
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
                    'num_classes': num_classes,
                    'class_names': class_names,
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
        
        # Update status to error
        update_training_status(model_name, {
            'status': 'error',
            'error': str(e)
        })
        
        return False


def update_training_status(model_name, status_update):
    """Update the training status for a model."""
    try:
        # Get path to status file
        status_dir = os.path.join(get_model_path(), 'training_status')
        os.makedirs(status_dir, exist_ok=True)
        status_file = os.path.join(status_dir, f"{model_name}.json")
        
        # If file exists, read and update
        status = {
            'model_name': model_name,
            'started_at': str(time.time()),
            'status': 'initializing',
            'progress': 0
        }
        
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    existing = json.load(f)
                    # Only update if valid
                    if isinstance(existing, dict):
                        status.update(existing)
            except (json.JSONDecodeError, ValueError) as e:
                # File exists but is corrupt - log and continue with new status
                logger.error(f"Error reading status file {status_file}, creating new status: {e}")
                # Maybe backup the corrupt file for debugging
                if os.path.getsize(status_file) > 0:
                    corrupt_file = f"{status_file}.corrupt"
                    try:
                        import shutil
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
        
        # Write updated status to file - use atomic write pattern to prevent corruption
        temp_file = f"{status_file}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(safe_status, f, indent=2)
        
        # Rename temp file to actual file (atomic operation)
        import os
        if os.name == 'nt':  # Windows
            # Windows requires removing the destination file first
            if os.path.exists(status_file):
                os.remove(status_file)
        os.rename(temp_file, status_file)
            
        logger.debug(f"Updated training status for {model_name}: {status_update.get('status')}")
    except Exception as e:
        logger.error(f"Error updating training status: {e}")


def get_training_status(model_name):
    """Get the current training status for a model."""
    try:
        status_file = os.path.join(get_model_path(), 'training_status', f"{model_name}.json")
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                return json.load(f)
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
            elif hasattr(obj, '__dict__'):
                # Handle class instances
                return str(obj)
            else:
                # Return basic types as is
                return obj
                
        # Process the entire status dictionary
        serializable_status = make_serializable(status)
        return serializable_status
        
    except Exception as e:
        logger.error(f"Error preparing status for JSON: {e}")
        # Return a simplified version that will definitely serialize
        return {
            'status': status_dict.get('status', 'error'),
            'message': status_dict.get('message', 'Error preparing status') 
        } 