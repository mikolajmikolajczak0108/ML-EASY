import os
import time
import json
import logging
import traceback
import threading
import pickle
import random
from pathlib import Path

from .utils import get_model_path, get_dataset_path

# Initialize logger
logger = logging.getLogger(__name__)


def train_model_task(model_name, dataset_name, architecture, epochs, batch_size, 
                    learning_rate, data_augmentation):
    """Background task to train a machine learning model with fastai/PyTorch."""
    try:
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
            'message': f'Downloading pre-trained {architecture} model...'
        })
        
        logger.info(f"Building model architecture: {architecture}")
        
        # Map architecture string to fastai model
        try:
            if architecture == "ResNet50":
                model_arch = models.resnet50
            elif architecture == "MobileNetV2":
                model_arch = models.mobilenet_v2
            elif architecture == "EfficientNetB0":
                # Manual implementation required as fastai doesn't include EfficientNet directly
                model_arch = models.efficientnet_b0
            elif architecture == "VGG16":
                model_arch = models.vgg16
            elif architecture == "InceptionV3":
                model_arch = models.inception_v3
            elif architecture == "DenseNet121":
                model_arch = models.densenet121
            else:
                # Default to ResNet34 if architecture not recognized
                logger.warning(f"Architecture {architecture} not recognized, using ResNet34")
                model_arch = models.resnet34
                architecture = "ResNet34"
            
            update_training_status(model_name, {
                'status': 'preparing',
                'progress': 50,
                'stage': 'creating_model',
                'message': f'Creating {architecture} model for {num_classes} classes'
            })
            
            # Create fastai learner
            learn = vision_learner(
                dls, 
                model_arch, 
                metrics=[accuracy, error_rate, Precision(), Recall(), F1Score()],
                lr=learning_rate
            )
            
            update_training_status(model_name, {
                'status': 'preparing',
                'progress': 60,
                'stage': 'model_ready',
                'message': 'Model architecture ready for training'
            })
            
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
                self.training_start_time = time.time()
                
            def before_fit(self):
                self.training_start_time = time.time()
                update_training_status(self.model_name, {
                    'status': 'training',
                    'progress': 65,
                    'stage': 'before_training',
                    'epoch': 0,
                    'total_epochs': self.total_epochs,
                    'message': 'Preparing to start training...',
                    'metrics': {},
                    'started_at': self.training_start_time
                })
                
            def before_epoch(self):
                self.current_epoch = learn.epoch
                # Calculate progress - account for freeze and fine tune phases
                # First 1 epoch is freeze training at 5% each, remaining epochs are fine tuning
                base_progress = 65 # Starting progress for training phase
                freeze_progress = 5 # Progress per freeze epoch
                remaining_progress = 30 # Progress for all fine tune epochs
                
                if self.current_epoch == 0:
                    # First epoch (freeze) represents first 5% of progress
                    progress = base_progress + 2.5
                else:
                    # After freeze, divide remaining progress among fine tune epochs
                    fine_tune_progress = self.current_epoch * (remaining_progress / self.total_epochs)
                    progress = base_progress + freeze_progress + fine_tune_progress
                
                update_training_status(self.model_name, {
                    'status': 'training',
                    'progress': min(95, progress), # Cap at 95% until complete
                    'stage': 'training_epoch',
                    'phase': 'freeze' if self.current_epoch == 0 else 'fine_tune',
                    'epoch': self.current_epoch + 1,
                    'total_epochs': self.total_epochs,
                    'message': f'Training epoch {self.current_epoch + 1}/{self.total_epochs}',
                    'metrics': {}
                })
                
            def after_epoch(self):
                # Get current epoch metrics
                epoch = learn.epoch
                train_loss = float(learn.recorder.losses[-1])
                valid_loss = float(learn.recorder.values[-1][0])
                accuracy_val = float(learn.recorder.values[-1][1])
                error_rate_val = float(learn.recorder.values[-1][2])
                
                # Calculate progress percentage based on completed epochs
                base_progress = 65
                freeze_progress = 5
                remaining_progress = 30
                
                if epoch == 0:
                    # First epoch (freeze) is complete
                    progress = base_progress + freeze_progress
                else:
                    # Calculate fine-tuning progress
                    fine_tune_progress = (epoch + 1) * (remaining_progress / self.total_epochs)
                    progress = base_progress + freeze_progress + fine_tune_progress
                
                # Update training status with detailed metrics
                update_training_status(self.model_name, {
                    'status': 'training',
                    'progress': min(95, progress), # Cap at 95% until complete
                    'stage': 'epoch_complete',
                    'phase': 'freeze' if epoch == 0 else 'fine_tune',
                    'epoch': epoch + 1,
                    'total_epochs': self.total_epochs,
                    'message': f'Completed epoch {epoch + 1}/{self.total_epochs}',
                    'metrics': {
                        'loss': train_loss,
                        'accuracy': accuracy_val,
                        'val_loss': valid_loss,
                        'val_accuracy': 1 - error_rate_val,  # Convert error to accuracy
                        'epoch': epoch + 1,
                        'total_epochs': self.total_epochs
                    }
                })
                logger.info(f"Epoch {epoch+1}/{self.total_epochs} complete. Loss: {train_loss:.4f}, Accuracy: {accuracy_val:.4f}")
            
            def after_fit(self):
                # Update status when training is complete
                update_training_status(self.model_name, {
                    'status': 'saving',
                    'progress': 95,
                    'stage': 'training_complete',
                    'message': 'Training complete, saving model...',
                })
        
        # Register our custom callback
        status_cb = StatusCallback(model_name)
        learn.add_cb(status_cb)
        
        # Train the model
        learn.fine_tune(epochs, freeze_epochs=1)
        
        # Get final metrics from the recorder
        final_train_loss = float(learn.recorder.losses[-1])
        final_valid_loss = float(learn.recorder.values[-1][0])
        final_accuracy = float(learn.recorder.values[-1][1])
        final_error_rate = float(learn.recorder.values[-1][2])
        
        # 4. Save model
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
        
        # Save fastai model - this creates a proper pickle file
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
                'training_time': int(time.time() - float(status_cb.training_start_time)) if hasattr(status_cb, 'training_start_time') else None
            }
        })
        
        logger.info(f"Model training completed successfully: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error training model {model_name}: {e}")
        logger.error(traceback.format_exc())
        
        # Update status to error
        update_training_status(model_name, {
            'status': 'error',
            'error': str(e)
        })
        
        return False


def update_training_status(model_name, status):
    """Update and persist the training status for a model."""
    try:
        # Create status directory if it doesn't exist
        status_dir = os.path.join(get_model_path(), 'training_status')
        os.makedirs(status_dir, exist_ok=True)
        
        # Write status to file
        status_file = os.path.join(status_dir, f"{model_name}.json")
        with open(status_file, 'w') as f:
            json.dump(status, f)
            
        logger.debug(f"Updated training status for {model_name}: {status['status']}")
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