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
import math

from .utils import get_model_path, get_dataset_path
from .safe_json import safe_metrics, safe_json_value

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

def setup_model_logger(model_name):
    """
    Set up a logger specifically for a model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        logging.Logger: Logger instance
    """
    try:
        # Create logger for the model
        logger = logging.getLogger(f"model.{model_name}")
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
            
        # Create a directory for model logs if it doesn't exist
        model_log_dir = os.path.join(get_model_path(), 'logs')
        os.makedirs(model_log_dir, exist_ok=True)
        
        # Set up log file path
        log_file = os.path.join(model_log_dir, f"{model_name}.log")
        
        # Create and configure a file handler with explicit UTF-8 encoding
        try:
            # For Windows compatibility, use UTF-8 encoding with errors='replace'
            file_handler = logging.FileHandler(
                log_file, encoding='utf-8', errors='replace'
            )
        except TypeError:
            # Older Python versions don't support errors parameter
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            
        file_handler.setLevel(logging.INFO)
        
        # Create a custom formatter that handles Unicode safely
        class SafeFormatter(logging.Formatter):
            def format(self, record):
                # First apply the standard formatting
                formatted = super().format(record)
                
                # Replace Unicode characters that might cause issues
                if isinstance(formatted, str):
                    try:
                        # Try to keep the Unicode if possible
                        return formatted
                    except UnicodeEncodeError:
                        # Fall back to ASCII with replacements if needed
                        return formatted.encode('ascii', 'replace').decode('ascii')
                return formatted
                
        # Create and set formatters
        formatter = SafeFormatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        
        # Log start message
        logger.info(f"Model logger initialized for: {model_name}")
        
        return logger
        
    except Exception as e:
        app_logger = logging.getLogger(__name__)
        app_logger.error(f"Error setting up logger for model {model_name}: {e}")
        return logging.getLogger(__name__)  # Return default logger as fallback


# Custom PrintLogger class that captures print output to both console and log file
class PrintLogger:
    """Logger that both prints to stdout and logs to a file."""
    
    def __init__(self, terminal, model_logger):
        """
        Initialize with both terminal output and file logger.
        
        Args:
            terminal: Stream for terminal output
            model_logger: Logger instance for the model
        """
        self.terminal = terminal
        self.model_logger = model_logger
        
    def write(self, message):
        """
        Write message to terminal and logs.
        
        Args:
            message (str): Message to write
        """
        try:
            # Always write to terminal
            self.terminal.write(message)
            
            # Only log non-empty messages
            message = message.strip()
            if message:
                # Check if this is a progress bar (contains block characters)
                has_block_chars = any(char in message for char in 
                                    ['█', '▁', '▂', '▃', '▄'])
                if has_block_chars:
                    # Convert Unicode progress bar to ASCII for safer logging
                    ascii_message = message
                    block_chars = ['█', '▁', '▂', '▃', '▄', '▅', '▆', '▇']
                    for char in block_chars:
                        ascii_message = ascii_message.replace(char, '#')
                    self.model_logger.debug(ascii_message)
                else:
                    # Regular message
                    self.model_logger.info(message)
        except UnicodeEncodeError:
            # Handle Unicode encoding error by falling back to ASCII
            try:
                ascii_message = message.encode('ascii', 'replace').decode('ascii')
                self.model_logger.info(ascii_message)
            except Exception:
                # Last resort: just log that we received a message
                self.model_logger.debug("Progress update (Unicode characters removed)")
        except Exception as e:
            # Catch other exceptions to prevent training from breaking
            self.model_logger.warning(f"Error in logger: {str(e)}")
    
    def flush(self):
        """Flush the terminal stream."""
        self.terminal.flush()


# StatusCallback class moved outside of train_model_task to make it picklable
class StatusCallback:
    """Custom callback to update training status."""
    
    def __init__(self, model_name):
        """Initialize the callback with model information."""
        self.model_name = model_name
        self.name = 'StatusCallback'  # Required by fastai
        self.current_epoch = 0
        self.total_epochs = 0  # Will be set based on freeze+fine_tune
        self.training_start_time = None
        self.epoch_start_time = None
        self.learn = None
        self.freeze_phase_complete = False
        self.actual_total_epochs = 0  # This will be set with the proper value later
        self.last_update_time = time.time()
        self.batch_counter = 0
        self.reported_stalled = False
        
    def __call__(self, event_name):
        """Handle callback events."""
        try:
            # Add more debug information
            print(f"Callback event: {event_name}")
            
            # Send intermediate status update every 15 seconds to prevent "stalled" detection
            current_time = time.time()
            if current_time - self.last_update_time > 15:  # Update every 15 seconds even during batch processing
                self._send_interim_update()
                self.last_update_time = current_time
            
            # Delegate to specific event handlers
            if event_name == 'before_fit':
                self.before_fit()
            elif event_name == 'before_epoch':
                self.before_epoch()
            elif event_name == 'after_epoch':
                self.after_epoch()
            elif event_name == 'after_fit':
                self.after_fit()
            elif event_name == 'after_batch':
                self.after_batch()
        except Exception as e:
            # Catch and log any errors in the callback
            logger.error(f"Error in StatusCallback event {event_name}: {e}")
            print(f"Error in callback ({event_name}): {e}")
    
    def _send_interim_update(self):
        """Send intermediate status updates during batch processing to prevent stalled detection."""
        try:
            # Calculate current progress 
            if self.freeze_phase_complete:
                # Fine-tune phase
                epoch_message = f"{self.current_epoch+1}/{self.actual_total_epochs-1} (fine-tune)"
                overall_epoch = self.current_epoch + 2  # 1 for freeze + current fine-tune epoch + 1 (0-indexed)
            else:
                # Freeze phase
                epoch_message = f"{self.current_epoch+1}/1 (freeze)"
                overall_epoch = 1  # Always 1 in freeze phase (1-indexed)
            
            # Calculate progress percentage
            base_progress = 65
            freeze_progress = 5
            remaining_progress = 30
            
            if not self.freeze_phase_complete:
                progress = base_progress + (freeze_progress * (self.current_epoch + 1)) + \
                           (self.batch_counter / (self.learn.dls.train_ds.__len__() / self.learn.dls.train.bs)) * 5
            else:
                # Fine-tune phase
                fine_tune_progress = (self.current_epoch + 1) * (remaining_progress / max(1, self.actual_total_epochs - 1))
                progress = base_progress + freeze_progress + fine_tune_progress

            # Send a heartbeat update to keep the frontend aware that training is alive
            phase = 'freeze' if not self.freeze_phase_complete else 'fine_tune'
            update_training_status(self.model_name, {
                'status': 'training',
                'progress': min(95, progress),
                'stage': 'processing_batches',
                'phase': phase,
                'epoch': overall_epoch,
                'total_epochs': self.actual_total_epochs,
                'message': f'Training epoch {overall_epoch}/{self.actual_total_epochs} ({phase} phase) - processing batches',
                'last_update': str(time.time()),  # Add timestamp to help frontend detect if updates are stalled
                'batch_counter': self.batch_counter
            })
            print(f"Sent interim update during batch processing - Epoch: {overall_epoch}, Progress: {progress:.1f}%")
        except Exception as e:
            print(f"Error sending interim update: {e}")
    
    def after_batch(self):
        """Handle after_batch event to update progress during long epochs."""
        try:
            self.batch_counter += 1
            # Only send update every 10 batches to avoid flooding
            if self.batch_counter % 10 == 0:
                self._send_interim_update()
        except Exception as e:
            print(f"Error in after_batch: {e}")
    
    def before_fit(self):
        self.training_start_time = time.time()
        self.last_update_time = time.time()
        self.batch_counter = 0
        self.reported_stalled = False
        try:
            print("\n>> Training starting - before_fit callback")
            # Set total epochs based on the learner's n_epoch attribute if available
            try:
                if hasattr(self.learn, 'n_epoch'):
                    # n_epoch typically only includes the current phase (freeze or fine_tune)
                    # We need to account for both phases (freeze + fine_tune)
                    detected_epochs = self.learn.n_epoch
                    # If we're in freeze phase, add the fine_tune epochs
                    # If we're in fine_tune phase, keep the current value
                    if not hasattr(self, 'total_epochs') or self.total_epochs == 0:
                        self.total_epochs = detected_epochs
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
                'training_start': time.time(),
                'last_update': str(time.time())
            })
            print("Training status updated: starting_training")
        except Exception as e:
            logger.error(f"Error updating status in before_fit: {e}")
            print(f"Error in before_fit: {e}")
    
    def before_epoch(self):
        try:
            # Reset batch counter for the new epoch
            self.batch_counter = 0
            self.last_update_time = time.time()
            
            # Be defensive about accessing learn.epoch
            epoch = 0
            try:
                if hasattr(self.learn, 'epoch'):
                    epoch = self.learn.epoch
                self.current_epoch = epoch
                self.epoch_start_time = time.time()
                
                # Check if we've transitioned from freeze to fine_tune phase
                # In fastai, fine_tune first does one epoch of freeze, then switches to fine_tune
                # So when epoch resets to 0 after being > 0, we've switched phases
                if epoch == 0 and self.current_epoch > 0:
                    self.freeze_phase_complete = True
                    print("\n>> Transitioning from freeze to fine-tune phase")
                    
            except Exception as e:
                print(f"Error getting epoch information: {e}")
                pass
            
            # Calculate the total progress across both phases
            if self.freeze_phase_complete:
                # We're in fine-tune phase now
                epoch_message = f"{epoch+1}/{self.actual_total_epochs-1} (fine-tune)"
                overall_epoch = epoch + 2  # 1 for freeze + current fine-tune epoch + 1 (0-indexed)
            else:
                # We're still in freeze phase
                epoch_message = f"{epoch+1}/1 (freeze)"
                overall_epoch = 1  # Always 1 in freeze phase (1-indexed)
            
            print(f"\n>> Starting epoch {epoch_message} (overall: {overall_epoch}/{self.actual_total_epochs})")
            
            # Calculate progress
            base_progress = 65
            freeze_progress = 5 
            remaining_progress = 30
            
            if not self.freeze_phase_complete:
                # Freeze phase (first epoch)
                progress = base_progress + (freeze_progress * (epoch + 1))
            else:
                # Fine-tune phase (remaining epochs)
                fine_tune_progress = (epoch + 1) * (remaining_progress / max(1, self.actual_total_epochs - 1))
                progress = base_progress + freeze_progress + fine_tune_progress
            
            phase = 'freeze' if not self.freeze_phase_complete else 'fine_tune'
            print(f"Training phase: {phase}, Progress: {progress:.1f}%")
            print(f"Starting batch processing...")
            
            update_training_status(self.model_name, {
                'status': 'training',
                'progress': min(95, progress),
                'stage': 'training_epoch',
                'phase': phase,
                'epoch': overall_epoch,
                'total_epochs': self.actual_total_epochs,
                'message': f'Training epoch {overall_epoch}/{self.actual_total_epochs} ({phase} phase)',
                'metrics': {},
                'last_update': str(time.time())
            })
        except Exception as e:
            logger.error(f"Error updating status in before_epoch: {e}")
            print(f"Error in before_epoch: {e}")
    
    def after_epoch(self):
        try:
            # Reset stalled flag with each successful epoch completion
            self.reported_stalled = False
            
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
            
            # Format the completed epoch message consistently
            if self.freeze_phase_complete:
                epoch_message = f"{epoch+1}/{self.actual_total_epochs-1} (fine-tune)"
                overall_epoch = epoch + 2  # freeze(1) + current epoch + 1 (0-indexed)
            else:
                epoch_message = f"{epoch+1}/1 (freeze)"
                overall_epoch = 1  # First epoch of freeze phase (1-indexed)
                
            print(f"\n>> Completed epoch {epoch_message} (overall: {overall_epoch}/{self.actual_total_epochs}) in {epoch_duration:.2f} seconds")
            
            # Safely extract metrics
            try:
                if hasattr(self.learn, 'recorder') and hasattr(self.learn.recorder, 'losses') and len(self.learn.recorder.losses) > 0:
                    try:
                        # Ensure we're dealing with valid numbers
                        loss_value = self.learn.recorder.losses[-1]
                        train_loss = float(loss_value) if loss_value is not None and not math.isnan(loss_value) else 0
                    except (ValueError, TypeError):
                        train_loss = 0
                    
                if hasattr(self.learn, 'recorder') and hasattr(self.learn.recorder, 'values') and len(self.learn.recorder.values) > 0:
                    values = self.learn.recorder.values[-1]
                    if len(values) > 0:
                        try:
                            val_loss_value = values[0]
                            valid_loss = float(val_loss_value) if val_loss_value is not None and not math.isnan(val_loss_value) else 0
                        except (ValueError, TypeError):
                            valid_loss = 0
                            
                    if len(values) > 1:
                        try:
                            acc_value = values[1]
                            accuracy_val = float(acc_value) if acc_value is not None and not math.isnan(acc_value) else 0
                        except (ValueError, TypeError):
                            accuracy_val = 0
                            
                    if len(values) > 2:
                        try:
                            err_value = values[2]
                            error_rate_val = float(err_value) if err_value is not None and not math.isnan(err_value) else 0
                        except (ValueError, TypeError):
                            error_rate_val = 0
                            
                print(f"Metrics - Train loss: {train_loss:.4f}, Validation loss: {valid_loss:.4f}")
                print(f"Accuracy: {accuracy_val:.4f} ({accuracy_val*100:.2f}%), Error rate: {error_rate_val:.4f}")
                print(f"Learning rate: {self.learn.opt.hypers[-1]['lr'] if hasattr(self.learn, 'opt') and hasattr(self.learn.opt, 'hypers') else 'unknown'}")
                
            except Exception as metric_err:
                logger.error(f"Error extracting metrics: {metric_err}")
                print(f"Error extracting metrics: {metric_err}")
            
            # Calculate progress
            base_progress = 65
            freeze_progress = 5
            remaining_progress = 30
            
            # Calculate overall progress based on current phase and epoch
            if not self.freeze_phase_complete:
                progress = base_progress + freeze_progress
            else:
                # In fine-tune phase
                current_fine_tune_epoch = epoch + 1  # 1-indexed epoch within fine-tune
                total_fine_tune_epochs = self.actual_total_epochs - 1  # Subtract 1 for freeze epoch
                fine_tune_progress = current_fine_tune_epoch * (remaining_progress / max(1, total_fine_tune_epochs))
                progress = base_progress + freeze_progress + fine_tune_progress
            
            print(f"Training progress: {min(95, progress):.1f}%")
            
            # Calculate total training time so far
            total_time = 0
            if self.training_start_time:
                total_time = time.time() - self.training_start_time
                time_per_epoch = total_time / overall_epoch
                remaining_epochs = self.actual_total_epochs - overall_epoch
                estimated_remaining = time_per_epoch * remaining_epochs
                
                print(f"Total training time: {format_duration(total_time)}")
                print(f"Estimated time remaining: {format_duration(estimated_remaining)}")
            
            # Update training status with detailed metrics
            epoch_metrics = {
                'loss': train_loss,
                'accuracy': accuracy_val,
                'val_loss': valid_loss,
                'val_accuracy': 1 - error_rate_val,
                'epoch': overall_epoch,
                'total_epochs': self.actual_total_epochs,
                'epoch_duration': epoch_duration,
                'total_duration': total_time
            }
            
            # Save this epoch's metrics to our training history
            save_training_history(self.model_name, epoch_metrics)
            
            update_training_status(self.model_name, {
                'status': 'training',
                'progress': min(95, progress),
                'stage': 'epoch_complete',
                'phase': 'freeze' if not self.freeze_phase_complete else 'fine_tune',
                'epoch': overall_epoch, 
                'total_epochs': self.actual_total_epochs,
                'message': f'Completed epoch {overall_epoch}/{self.actual_total_epochs}',
                'metrics': epoch_metrics,
                'last_update': str(time.time())
            })
            logger.info(f"Epoch {overall_epoch}/{self.actual_total_epochs} complete. Loss: {train_loss:.4f}, Accuracy: {accuracy_val:.4f}")
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
                'total_duration': total_time,
                'last_update': str(time.time())
            })
        except Exception as e:
            logger.error(f"Error updating status in after_fit: {e}")
            print(f"Error in after_fit: {e}")


def save_training_history(model_name, epoch_metrics):
    """
    Save training history metrics for each epoch to a JSON file.
    This allows reviewing training history even after training is complete.
    
    Args:
        model_name (str): Name of the model
        epoch_metrics (dict): Metrics for the current epoch
    """
    try:
        # Get the path to save training history
        history_dir = os.path.join(get_model_path(), 'training_history')
        os.makedirs(history_dir, exist_ok=True)
        history_file = os.path.join(history_dir, f"{model_name}.json")
        
        # Get existing history if available
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    existing = json.load(f)
                    if isinstance(existing, list):
                        history = existing
            except Exception as e:
                logger.error(f"Error reading training history: {e}")
        
        # Make metrics JSON-serializable
        safe_metrics = safe_json_value(epoch_metrics)
        safe_metrics['timestamp'] = time.time()  # Add timestamp
        
        # Append new metrics
        history.append(safe_metrics)
        
        # Save back to file
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
        logger.debug(f"Saved training history for {model_name}, epoch {epoch_metrics.get('epoch', 'unknown')}")
    except Exception as e:
        logger.error(f"Error saving training history: {e}")
        logger.error(traceback.format_exc())


def get_training_history(model_name):
    """
    Get the complete training history for a model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        list: List of epoch metrics in chronological order
    """
    try:
        history_file = os.path.join(get_model_path(), 'training_history', f"{model_name}.json")
        if not os.path.exists(history_file):
            return []
            
        with open(history_file, 'r') as f:
            history = json.load(f)
            return history if isinstance(history, list) else []
    except Exception as e:
        logger.error(f"Error getting training history: {e}")
        return []


def fine_tune_model(model_name, new_model_name, epochs=5, learning_rate=0.0001, batch_size=16, augmentation=True):
    """
    Start fine-tuning of an existing trained model.
    
    Args:
        model_name (str): Name of the existing model to fine-tune
        new_model_name (str): Name for the fine-tuned model
        epochs (int): Number of fine-tuning epochs
        learning_rate (float): Learning rate for fine-tuning (should be lower than initial training)
        batch_size (int): Batch size for training
        augmentation (bool): Whether to use data augmentation
        
    Returns:
        dict: Status information about the started fine-tuning task
    """
    logger.info(f"Starting model fine-tuning: {model_name} → {new_model_name}")
    logger.info(f"Parameters: epochs={epochs}, lr={learning_rate}, batch_size={batch_size}")
    
    try:
        # Check if ML libraries are imported
        if not LIBRARIES_IMPORTED:
            raise ImportError("Required ML libraries are not available")
        
        # Check if source model exists
        model_path = os.path.join(get_model_path(), 'saved_models', model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Source model {model_name} not found at {model_path}")
            
        # Get model metadata to find what dataset was used
        metadata_file = os.path.join(model_path, 'metadata.json')
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Model metadata not found for {model_name}")
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            dataset_name = metadata.get('dataset')
            architecture = metadata.get('architecture', 'resnet34')
            
        if not dataset_name:
            raise ValueError(f"Dataset information missing from model metadata")
            
        # Check if dataset still exists
        dataset_path = os.path.join(get_dataset_path(), dataset_name)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")
        
        # Create directory for the new model
        new_model_dir = os.path.join(get_model_path(), 'saved_models', new_model_name)
        os.makedirs(new_model_dir, exist_ok=True)
        
        # Set initial status
        update_training_status(new_model_name, {
            'status': 'preparing',
            'progress': 0,
            'dataset_name': dataset_name,
            'architecture': architecture,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'fine_tuned_from': model_name,
            'message': 'Preparing fine-tuning environment',
            'last_update': str(time.time())
        })
        
        # Start fine-tuning in a background thread
        training_thread = threading.Thread(
            target=fine_tune_model_task,
            args=(
                model_name, new_model_name, dataset_name, epochs, learning_rate,
                batch_size, augmentation
            )
        )
        training_thread.daemon = True
        training_thread.start()
        
        return {
            'status': 'started',
            'model_name': new_model_name,
            'fine_tuned_from': model_name,
            'thread_id': training_thread.ident
        }
        
    except Exception as e:
        error_message = f"Error starting model fine-tuning: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        
        # Update status to error
        update_training_status(new_model_name, {
            'status': 'error',
            'progress': 0,
            'message': error_message,
            'last_update': str(time.time())
        })
        
        return {
            'status': 'error',
            'message': error_message
        }


def fine_tune_model_task(source_model_name, new_model_name, dataset_name, epochs=5, 
                      learning_rate=0.0001, batch_size=16, augmentation=True):
    """
    Background task for model fine-tuning.
    
    Args:
        source_model_name (str): Name of the existing model to fine-tune
        new_model_name (str): Name for the fine-tuned model
        dataset_name (str): Name of the dataset to use
        epochs (int): Number of fine-tuning epochs
        learning_rate (float): Learning rate for fine-tuning
        batch_size (int): Batch size
        augmentation (bool): Whether to use data augmentation
    """
    # Set up logging for this task
    model_logger = setup_model_logger(new_model_name)
    
    # Capture print output to log file
    original_stdout = sys.stdout
    sys.stdout = PrintLogger(sys.stdout, model_logger)
    
    try:
        # Mark the start time
        start_time = time.time()
        print(f"Starting fine-tuning process: {source_model_name} → {new_model_name}")
        print(f"Using dataset: {dataset_name}")
        print(f"Fine-tuning parameters: epochs={epochs}, learning_rate={learning_rate}, batch_size={batch_size}")
        
        # Import required libraries
        try:
            import torch
            from fastai.vision.all import (
                ImageDataLoaders, load_learner, Resize, aug_transforms, 
                RandomResizedCrop, Normalize, ClassificationInterpretation
            )
            from fastai.callback.all import SaveModelCallback, EarlyStoppingCallback, ReduceLROnPlateau
        except ImportError as e:
            print(f"Failed to import required libraries: {e}")
            update_training_status(new_model_name, {
                'status': 'error',
                'progress': 0,
                'message': f'Error importing ML libraries: {str(e)}',
                'last_update': str(time.time())
            })
            return
        
        # Update status
        update_training_status(new_model_name, {
            'status': 'preparing',
            'progress': 10,
            'message': 'Loading source model...',
            'last_update': str(time.time())
        })
        
        # Load the source model
        try:
            source_model_path = os.path.join(get_model_path(), 'saved_models', source_model_name)
            model_pkl_path = os.path.join(source_model_path, 'model.pkl')
            export_pkl_path = os.path.join(source_model_path, 'export.pkl')
            
            # Try different model files
            if os.path.exists(export_pkl_path):
                print(f"Loading model from {export_pkl_path}")
                learn = load_learner(export_pkl_path)
            elif os.path.exists(model_pkl_path):
                print(f"Loading model from {model_pkl_path}")
                learn = load_learner(model_pkl_path)
            else:
                raise FileNotFoundError(f"No model file found for {source_model_name}")
                
            print(f"Model loaded successfully with {len(learn.dls.vocab)} classes")
        except Exception as e:
            error_msg = f"Failed to load source model: {str(e)}"
            print(f"ERROR: {error_msg}")
            update_training_status(new_model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'model_loading_error',
                'last_update': str(time.time())
            })
            return
        
        # Update status
        update_training_status(new_model_name, {
            'status': 'preparing',
            'progress': 30,
            'message': 'Preparing dataset for fine-tuning...',
            'last_update': str(time.time())
        })
        
        # Get dataset path
        dataset_path = os.path.join(get_dataset_path(), dataset_name)
        print(f"Dataset path: {dataset_path}")
        
        # Create a fresh DataLoader with the same dataset
        # This ensures we're using the latest data even if the dataset has changed
        try:
            from pathlib import Path
            path = Path(dataset_path)
            
            # Set up augmentations for fine-tuning
            if augmentation:
                # Use slightly stronger augmentation for fine-tuning
                aug_tfms = aug_transforms(
                    max_rotate=30.0,
                    max_zoom=1.2,
                    max_lighting=0.4,
                    max_warp=0.3,
                    p_affine=0.8,
                    p_lighting=0.8
                )
                
                tfms = [
                    RandomResizedCrop(224, min_scale=0.5),
                    *aug_tfms
                ]
                print("Using enhanced augmentation for fine-tuning")
            else:
                tfms = [Resize(224)]
                print("Using minimal transformations: Resize(224)")
            
            # ImageNet stats for normalization
            stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
            # Create new DataLoaders
            dls = ImageDataLoaders.from_folder(
                path,
                valid_pct=0.2,
                seed=42,
                bs=batch_size,
                item_tfms=Resize(224),
                batch_tfms=[*tfms, Normalize.from_stats(*stats)],
                num_workers=0
            )
            
            # Replace the DataLoaders in the learner
            learn.dls = dls
            print(f"Created fresh DataLoaders with batch size: {batch_size}")
            print(f"Classes: {learn.dls.vocab}")
        except Exception as e:
            error_msg = f"Failed to create DataLoaders: {str(e)}"
            print(f"ERROR: {error_msg}")
            update_training_status(new_model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'dataloader_creation_error',
                'last_update': str(time.time())
            })
            return
        
        # Update status
        update_training_status(new_model_name, {
            'status': 'training',
            'progress': 50,
            'message': 'Starting fine-tuning...',
            'last_update': str(time.time())
        })
        
        # Set up callbacks
        status_cb = StatusCallback(new_model_name)
        status_cb.actual_total_epochs = epochs + 1  # freeze(1) + fine_tune(epochs)
        
        # Add callbacks for better training
        learn.add_cb(status_cb)
        learn.add_cb(SaveModelCallback(monitor='accuracy', comp=max, fname='best_model'))
        learn.add_cb(EarlyStoppingCallback(monitor='accuracy', patience=3, min_delta=0.01))
        learn.add_cb(ReduceLROnPlateau(monitor='valid_loss', patience=2, factor=0.5, min_delta=0.01))
        
        # Set a lower learning rate for fine-tuning (typically 10x smaller than initial training)
        learn.lr = learning_rate
        
        # Fine-tune the model
        try:
            print("Setting discriminative learning rates for fine-tuning...")
            
            # First freeze and train just the head for 1 epoch
            learn.freeze()
            print(f"Training only the head for 1 epoch with lr={learning_rate}")
            learn.fit_one_cycle(1, learning_rate)
            
            # Then unfreeze and train the entire model with discriminative learning rates
            learn.unfreeze()
            
            # Use discriminative learning rates - very low for early layers
            lr_max = learning_rate
            lr_min = lr_max / 20  # Even lower for early layers in fine-tuning
            print(f"Using discriminative learning rates from {lr_min:.1e} to {lr_max:.1e}")
            
            print(f"Fine-tuning full model for {epochs} epochs")
            learn.fit_one_cycle(epochs, slice(lr_min, lr_max))
            
            # Get final metrics
            final_metrics = {}
            if hasattr(learn, 'recorder') and len(learn.recorder.values) > 0:
                values = learn.recorder.values[-1]
                if len(values) > 0:
                    final_metrics['val_loss'] = float(values[0])
                if len(values) > 1:
                    final_metrics['accuracy'] = float(values[1])
                if len(values) > 2:
                    final_metrics['error_rate'] = float(values[2])
            
            print(f"Fine-tuning complete. Final accuracy: {final_metrics.get('accuracy', 'unknown')}")
            
            # Generate confusion matrix
            try:
                print("\nGenerating confusion matrix on validation set...")
                interp = ClassificationInterpretation.from_learner(learn)
                print(f"Confusion matrix:\n{interp.confusion_matrix()}")
                print("\nMost confused classes:")
                print(interp.most_confused(min_val=1))
            except Exception as cm_err:
                print(f"Could not generate confusion matrix: {cm_err}")
        except Exception as train_err:
            error_msg = f"Error during fine-tuning: {str(train_err)}"
            print(f"ERROR: {error_msg}")
            update_training_status(new_model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'fine_tuning_error',
                'last_update': str(time.time())
            })
            return
        
        # Save the fine-tuned model
        try:
            update_training_status(new_model_name, {
                'status': 'saving',
                'progress': 95,
                'message': 'Saving fine-tuned model...',
                'metrics': final_metrics,
                'last_update': str(time.time())
            })
            
            # Create model directory
            model_dir = os.path.join(get_model_path(), 'saved_models', new_model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model metadata
            metadata = {
                'model_name': new_model_name,
                'fine_tuned_from': source_model_name,
                'dataset': dataset_name,
                'architecture': learn.model.__class__.__name__,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'data_augmentation': augmentation,
                'date_created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_classes': len(learn.dls.vocab),
                'class_names': list(learn.dls.vocab),
                'final_metrics': final_metrics
            }
            
            with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Save the model
            model_pkl_path = os.path.join(model_dir, 'model.pkl')
            learn.export(model_pkl_path)
            
            # Also save as export.pkl for compatibility
            export_pkl_path = os.path.join(model_dir, 'export.pkl')
            learn.export(export_pkl_path)
            
            print(f"Fine-tuned model saved to: {model_dir}")
            
            # Mark as completed
            update_training_status(new_model_name, {
                'status': 'completed',
                'progress': 100,
                'message': 'Fine-tuning completed successfully!',
                'metrics': final_metrics,
                'model_info': {
                    'model_name': new_model_name,
                    'fine_tuned_from': source_model_name,
                    'dataset': dataset_name,
                    'architecture': learn.model.__class__.__name__,
                    'num_classes': len(learn.dls.vocab),
                    'class_names': list(learn.dls.vocab)
                },
                'last_update': str(time.time())
            })
            
            return True
        except Exception as save_err:
            error_msg = f"Error saving fine-tuned model: {str(save_err)}"
            print(f"ERROR: {error_msg}")
            update_training_status(new_model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'save_error',
                'last_update': str(time.time())
            })
            return False
    except Exception as e:
        logger.error(f"Error in fine-tuning task: {e}")
        logger.error(traceback.format_exc())
        update_training_status(new_model_name, {
            'status': 'error',
            'error': str(e),
            'last_update': str(time.time())
        })
        return False
    finally:
        # Restore stdout
        sys.stdout = original_stdout


def train_model(dataset_name, model_name, architecture='resnet34', epochs=10, learning_rate=0.001, batch_size=16, validation_pct=0.2, pretrained=True, img_size=224, augmentation=True):
    """
    Start the training of a deep learning model on a dataset.
    
    Args:
        dataset_name (str): Name of the dataset to use for training
        model_name (str): Name to give the trained model
        architecture (str): Neural network architecture to use (e.g., resnet34)
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for the optimizer
        batch_size (int): Batch size for training
        validation_pct (float): Percentage of data to use for validation
        pretrained (bool): Whether to use pretrained weights
        img_size (int): Size to resize images to
        augmentation (bool): Whether to use data augmentation
        
    Returns:
        dict: Status information about the started training task
    """
    logger.info(f"Starting model training for {model_name} using {architecture} on {dataset_name}")
    logger.info(f"Parameters: epochs={epochs}, lr={learning_rate}, batch_size={batch_size}")
    logger.info(f"Image size: {img_size}, validation: {validation_pct}, pretrained: {pretrained}")
    
    try:
        # Check if ML libraries are imported
        if not LIBRARIES_IMPORTED:
            raise ImportError("Required ML libraries are not available")
        
        # Get dataset path
        dataset_path = os.path.join(get_dataset_path(), dataset_name)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")
        
        # Get model directory
        model_dir = os.path.join(get_model_path(), 'saved_models', model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Check dataset size
        dataset_size = count_dataset_images(dataset_path)
        logger.info(f"Dataset size: {dataset_size} images")
        
        # Get custom parameters based on dataset size - import from utils
        try:
            from ...utils.ml_helpers import get_augmentation_parameters, get_training_parameters
            
            # Get augmentation and training parameters based on dataset size
            if dataset_size < 100:
                logger.info(f"Small dataset detected ({dataset_size} images). Adjusting parameters.")
                
                # Get optimized parameters for small datasets
                aug_params = get_augmentation_parameters(dataset_size)
                train_params = get_training_parameters(dataset_size)
                
                # Override with optimized parameters if small dataset
                if augmentation:
                    logger.info(f"Using enhanced augmentation for small dataset: {aug_params}")
                    # We'll use these parameters in the fastai transforms
                
                # Apply training parameter overrides
                learning_rate = train_params.get('learning_rate', learning_rate)
                batch_size = train_params.get('batch_size', batch_size)
                epochs = train_params.get('epochs', epochs)
                
                logger.info(f"Adjusted parameters: epochs={epochs}, lr={learning_rate}, batch_size={batch_size}")
        except ImportError:
            logger.warning("Could not import augmentation and training parameter helpers. Using defaults.")
        
        # Set initial status
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 0,
            'dataset_name': dataset_name,
            'architecture': architecture,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'img_size': img_size,
            'augmentation': augmentation,
            'message': 'Preparing training environment'
        })
        
        # Start training in a background thread
        training_thread = threading.Thread(
            target=train_model_task,
            args=(
                dataset_name, model_name, architecture, epochs, learning_rate,
                batch_size, validation_pct, pretrained, img_size, augmentation, dataset_size
            )
        )
        training_thread.daemon = True
        training_thread.start()
        
        return {
            'status': 'started',
            'model_name': model_name,
            'thread_id': training_thread.ident
        }
        
    except Exception as e:
        error_message = f"Error starting model training: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        
        # Update status to error
        update_training_status(model_name, {
            'status': 'error',
            'progress': 0,
            'message': error_message
        })
        
        return {
            'status': 'error',
            'message': error_message
        }


def train_model_task(dataset_name, model_name, architecture='resnet34', epochs=10, 
                  learning_rate=0.001, batch_size=16, validation_pct=0.2, 
                  pretrained=True, img_size=224, augmentation=True, dataset_size=None):
    """
    Background task for model training.
    
    Args:
        dataset_name (str): Name of the dataset
        model_name (str): Name for the trained model
        architecture (str): Architecture name
        epochs (int): Training epochs
        learning_rate (float): Learning rate
        batch_size (int): Batch size
        validation_pct (float): Validation percentage
        pretrained (bool): Use pretrained weights
        img_size (int): Image size
        augmentation (bool): Whether to use data augmentation
        dataset_size (int, optional): Total number of images in dataset
    """
    # Set up logging for this task
    model_logger = setup_model_logger(model_name)
    
    # Capture print output to log file
    original_stdout = sys.stdout
    sys.stdout = PrintLogger(sys.stdout, model_logger)
    
    try:
        # Mark the start time
        start_time = time.time()
        print(f"Starting training process for model: {model_name}")
        print(f"Using dataset: {dataset_name}, architecture: {architecture}")
        print(f"Training parameters: epochs={epochs}, learning_rate={learning_rate}, batch_size={batch_size}")
        
        try:
            # Import libraries within the task to ensure they're available in the thread
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
            
            # Import fastai libraries
            from fastai.vision.all import (
                ImageDataLoaders, 
                Resize, 
                vision_learner, 
                error_rate, 
                accuracy,
                RandomResizedCrop,
                aug_transforms,
                Normalize,
                ClassificationInterpretation
            )
            from fastai.metrics import Precision, Recall, F1Score
            
            # Import torchvision models (required by fastai)
            import torchvision.models as models
            
            # Import additional libraries for better training
            from fastai.callback.all import ReduceLROnPlateau, SaveModelCallback, EarlyStoppingCallback
            # Comment out the problematic import that's causing errors with fastai 2.7.19
            # from fastai.learner import minimum, steep, valley, slide
            
        except ImportError as e:
            print(f"Failed to import required libraries: {e}")
            update_training_status(model_name, {
                'status': 'error',
                'progress': 0,
                'message': f'Error importing ML libraries: {str(e)}'
            })
            return
        
        # Update status: verifying data
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 10,
            'message': 'Verifying dataset'
        })
        
        # Get dataset path
        dataset_path = os.path.join(get_dataset_path(), dataset_name)
        print(f"Dataset path: {dataset_path}")
        
        # Verify dataset path exists
        if not os.path.exists(dataset_path):
            error_msg = f"Dataset path does not exist: {dataset_path}"
            print(f"ERROR: {error_msg}")
            update_training_status(model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'dataset_not_found'
            })
            return
            
        # Verify dataset has at least two classes
        class_dirs = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))]
        print(f"Found {len(class_dirs)} classes: {', '.join(class_dirs)}")
        
        if len(class_dirs) < 2:
            error_msg = f"Dataset must have at least 2 classes, found {len(class_dirs)}"
            print(f"ERROR: {error_msg}")
            update_training_status(model_name, {
                'status': 'error',
                'error': error_msg,
                'error_type': 'insufficient_classes'
            })
            return
        
        # Verify each class has at least a few images
        min_images_per_class = 3
        empty_classes = []
        
        for class_dir in class_dirs:
            class_path = os.path.join(dataset_path, class_dir)
            image_files = [f for f in os.listdir(class_path) 
                          if os.path.isfile(os.path.join(class_path, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            if len(image_files) < min_images_per_class:
                empty_classes.append((class_dir, len(image_files)))
        
        if empty_classes:
            print(f"WARNING: Some classes have fewer than {min_images_per_class} images:")
            for class_name, count in empty_classes:
                print(f"  - {class_name}: {count} images")
                
            print("Training may not be effective with too few images per class")
        
        # Update status: preparing
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 20,
            'message': 'Loading dataset'
        })
        
        # Get custom augmentation parameters for small datasets
        try:
            if dataset_size < 100 and augmentation:
                from ...utils.ml_helpers import get_augmentation_parameters
                aug_params = get_augmentation_parameters(dataset_size)
                print(f"Using enhanced augmentation for small dataset: {aug_params}")
                
                # Create custom augmentation transform with stronger augmentation for small datasets
                aug_tfms = aug_transforms(
                    max_rotate=aug_params.get('rotation_range', 30.0),  # Increased rotation
                    max_zoom=aug_params.get('zoom_range', 1.3),  # Increased zoom
                    max_lighting=min(0.95, aug_params.get('brightness_range', (0.6, 0.95))[1] if isinstance(aug_params.get('brightness_range'), tuple) else 0.3),
                    max_warp=aug_params.get('shear_range', 0.3),  # Increased warping
                    p_affine=0.8,  # Higher probability of applying affine transforms
                    p_lighting=0.8,  # Higher probability of lighting adjustments
                    xtra_tfms=[]  # Additional transforms can be added here
                )
            else:
                # Default augmentation - still improved over the original
                aug_tfms = aug_transforms(
                    max_rotate=20.0,
                    max_zoom=1.2,
                    max_lighting=0.3,
                    max_warp=0.2,
                    p_affine=0.75,
                    p_lighting=0.75
                ) if augmentation else None
        except ImportError:
            print("Could not import augmentation helpers. Using default augmentation.")
            aug_tfms = aug_transforms() if augmentation else None
        except Exception as aug_error:
            print(f"Error setting up augmentation: {aug_error}. Using default.")
            aug_tfms = aug_transforms() if augmentation else None
            
        # Update status: creating data loaders
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 30,
            'message': 'Preparing data loaders'
        })
        
        # Set up augmentations based on user selection
        if augmentation:
            tfms = [
                RandomResizedCrop(img_size, min_scale=0.5),
                *aug_tfms
            ]
            print(f"Using data augmentation with: RandomResizedCrop, aug_transforms()")
        else:
            tfms = [Resize(img_size)]
            print(f"Using minimal transformations: Resize({img_size})")
            
        # Create DataLoaders
        print("\n===== CREATING DATA LOADERS =====")
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 40,
            'stage': 'creating_dataloaders',
            'message': 'Creating DataLoaders...'
        })
        
        print(f"Creating DataLoaders with batch size: {batch_size}")
        try:
            print(f"Creating Path object for dataset: {dataset_path}")
            path = Path(dataset_path)
            
            print(f"Data transformations: {'Using augmentations' if augmentation else 'Minimal transforms'}")
            print(f"Batch size: {batch_size}, Workers: 0 (avoiding multiprocessing)")
            
            # Calculate proper validation split based on dataset size
            # For small datasets, use a larger validation split to ensure enough validation samples
            if dataset_size and dataset_size < 100:
                adjusted_valid_pct = min(0.3, validation_pct + 0.05)  # Increase slightly for small datasets
                print(f"Using adjusted validation split of {adjusted_valid_pct:.2f} for small dataset")
            else:
                adjusted_valid_pct = validation_pct
            
            print("Calling ImageDataLoaders.from_folder()...")
            start_time = time.time()
            
            # Create ImageDataLoaders with normalized inputs
            # This is essential for using pretrained models - we need proper normalization
            stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
            
            dls = ImageDataLoaders.from_folder(
                path,
                valid_pct=adjusted_valid_pct,
                seed=42,
                bs=batch_size,
                item_tfms=Resize(img_size),
                batch_tfms=[*tfms, Normalize.from_stats(*stats)],  # Add normalization transform
                num_workers=0  # Avoid Windows multiprocessing issues
            )
            load_time = time.time() - start_time
            print(f"DataLoaders created in {load_time:.2f} seconds")
            
            print(f"DataLoaders created. Number of classes: {len(dls.vocab)}")
            print(f"Class names: {dls.vocab}")
            print(f"Total items - Train: {len(dls.train_ds)}, Validation: {len(dls.valid_ds)}")
            
            # Check for class imbalance
            class_counts = {}
            for i, (x, y) in enumerate(dls.train_ds):
                class_name = dls.vocab[y]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print("Class distribution in training set:")
            for class_name, count in class_counts.items():
                print(f"  - {class_name}: {count} images")
            
            # Detect and warn about severe class imbalance
            if class_counts:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                imbalance_ratio = max_count / max(min_count, 1)
                
                if imbalance_ratio > 5:
                    print(f"WARNING: Severe class imbalance detected (ratio {imbalance_ratio:.1f}). "
                          f"Consider balancing your dataset for better results.")
            
            update_training_status(model_name, {
                'status': 'preparing',
                'progress': 50,
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
            return
            
        # Create the model
        print("\n===== CREATING MODEL =====")
        update_training_status(model_name, {
            'status': 'preparing',
            'progress': 60,
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
            
            # Known compatibility issues with certain architectures on some environments
            known_problematic_archs = ['MobileNetV2', 'mobile', 'mobilenet_v2']
            
            if architecture not in arch_mapping:
                print(f"Warning: Unknown architecture '{architecture}', falling back to resnet18")
                architecture = 'resnet18'
            
            # If we're using a potentially problematic architecture, check connectivity first
            arch_fn = arch_mapping.get(architecture, models.resnet18)
            if architecture.lower() in [a.lower() for a in known_problematic_archs]:
                test_url = "https://download.pytorch.org/models/hub/checkpoints/"
                try:
                    import urllib.request
                    print(f"Testing connection to PyTorch servers for {architecture}...")
                    urllib.request.urlopen(test_url, timeout=10)
                    print("Connection test successful, will attempt to use requested architecture")
                except Exception as conn_err:
                    print(f"Warning: Cannot connect to PyTorch server and {architecture} is known to have compatibility issues")
                    print(f"Automatically falling back to ResNet18 which works better with cached/offline models")
                    architecture = 'resnet18'
                    arch_fn = models.resnet18
            
            print(f"Using model architecture function: {arch_fn.__name__}")
            
            # Create the learner
            print(f"Creating vision_learner with {len(dls.vocab)} classes")
            print("This might take a moment to download pre-trained weights...")
            
            start_time = time.time()
            try:
                # Add timeout protection to prevent indefinite hanging
                timeout_seconds = 120  # 2 minutes timeout
                
                # Define proper metrics for the classification task
                metrics = [
                    accuracy,  # Overall accuracy
                    error_rate,  # Error rate
                    Precision(average='macro'),  # Precision with macro averaging
                    Recall(average='macro'),  # Recall with macro averaging
                    F1Score(average='macro')  # F1 score with macro averaging
                ]
                
                # Set environment variable to cache models locally
                os.environ['TORCH_HOME'] = os.path.join(get_model_path(), 'torch_cache')
                print(f"TORCH_HOME set to: {os.environ['TORCH_HOME']}")
                
                # Ensure model cache directory exists
                os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)
                
                # Create the learner with proper metrics
                learn = vision_learner(
                    dls, 
                    arch_fn, 
                    metrics=metrics,
                    pretrained=pretrained
                )
                
                creation_time = time.time() - start_time
                print(f"Model created successfully in {creation_time:.2f} seconds")
                
                # Verify if we have a large enough dataset for proper training
                if dataset_size and dataset_size < 20 * len(dls.vocab):
                    print(f"WARNING: Small dataset ({dataset_size} images) for {len(dls.vocab)} classes.")
                    print("It's recommended to have at least 20 images per class for effective training.")
                    print("Consider adding more data or using more aggressive augmentation.")
                
            except Exception as model_err:
                error_msg = str(model_err)
                print(f"Error creating vision_learner with {arch_fn.__name__}: {error_msg}")
                print(f"Detailed error: {traceback.format_exc()}")
                
                print("Trying with simpler ResNet18 architecture...")
                
                # Fallback to resnet18 which usually works
                fallback_start = time.time()
                try:
                    learn = vision_learner(
                        dls, 
                        models.resnet18, 
                        metrics=[accuracy, error_rate],
                        pretrained=True
                    )
                    
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
                        return
            
            # Find optimal learning rate
            print("\n===== FINDING OPTIMAL LEARNING RATE =====")
            try:
                print("Running learning rate finder to determine optimal learning rate")
                # Update to use lr_find without the suggest_funcs parameter that's causing errors
                lr_finder = learn.lr_find()
                
                # Get the suggested learning rate
                try:
                    suggested_lr = lr_finder.valley()
                except:
                    try:
                        suggested_lr = lr_finder.steep() / 10
                    except:
                        suggested_lr = learning_rate  # Fall back to specified learning rate
                
                # If no good learning rate found, use a reasonable default
                if not suggested_lr or suggested_lr < 1e-5 or suggested_lr > 0.1:
                    suggested_lr = learning_rate  # Fall back to user-provided learning rate
                    print(f"No optimal learning rate found, using provided learning rate: {suggested_lr}")
                else:
                    print(f"Using optimal learning rate from LR finder: {suggested_lr}")
                    
                # Apply a slight discount to the suggested learning rate for stability
                final_lr = suggested_lr * 0.9
                print(f"Final learning rate after adjustment: {final_lr}")
                
                # Update the learner with the optimal learning rate
                learn.lr = final_lr
            except Exception as lr_err:
                print(f"Error finding optimal learning rate: {lr_err}")
                print("Using the provided learning rate instead")
                learn.lr = learning_rate
            
            update_training_status(model_name, {
                'status': 'preparing',
                'progress': 60,
                'stage': 'model_created',
                'message': f'Model created with architecture: {architecture}',
                'actual_architecture': arch_fn.__name__  # Add the actual architecture function name
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
            return
        
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
        print(f"Learning rate: {learn.lr}")
        
        # Register our custom callback
        status_cb = StatusCallback(model_name)
        status_cb.actual_total_epochs = epochs + 1  # freeze(1) + fine_tune(epochs) 
        learn.add_cb(status_cb)
        
        # Add callbacks for better training
        learn.add_cb(SaveModelCallback(monitor='accuracy', comp=max, fname='best_model'))
        learn.add_cb(EarlyStoppingCallback(monitor='accuracy', patience=5, min_delta=0.01))
        learn.add_cb(ReduceLROnPlateau(monitor='valid_loss', patience=2, factor=0.5, min_delta=0.01))
        
        # Train the model with additional error handling
        try:
            # Set discriminative learning rates for transfer learning
            # Use a lower learning rate for early layers (which contain more general features)
            # and higher learning rates for later layers (which are more task-specific)
            print("Setting discriminative learning rates...")
            try:
                learn.freeze()  # Freeze all layers first
                
                # Train only the head with a high learning rate to quickly adapt to our classes
                print(f"Training only the head for 1 epoch with lr={learn.lr}")
                learn.fit_one_cycle(1, learn.lr)
                
                # Now unfreeze and train with discriminative learning rates
                learn.unfreeze()
                
                # Set discriminative learning rates - lower for early layers, higher for later layers
                # This helps preserve general features while adapting specific features to our task
                lr_max = learn.lr
                lr_min = lr_max / 10  # 10x lower learning rate for earliest layers
                print(f"Using discriminative learning rates from {lr_min:.1e} to {lr_max:.1e}")
                
                # Train with one-cycle policy for better convergence
                print(f"Training full model for {epochs} epochs with discriminative learning rates")
                learn.fit_one_cycle(epochs, slice(lr_min, lr_max))
                
            except ValueError as ve:
                # Handle specific known errors
                if "Target is multiclass" in str(ve) or "average=" in str(ve):
                    print(f"Error in metrics during training: {ve}")
                    print("Attempting to remove problematic metrics and continue...")
                    
                    # Reset metrics to just the basics
                    learn.metrics = [accuracy, error_rate]
                    print("Metrics reset to basics: accuracy, error_rate")
                    
                    # Try again with only the basic metrics
                    try:
                        learn.freeze()
                        learn.fit_one_cycle(1, learn.lr)
                        learn.unfreeze()
                        learn.fit_one_cycle(epochs, slice(learn.lr/10, learn.lr))
                        print("Training succeeded with basic metrics")
                    except Exception as retry_err:
                        print(f"Training still failed after metric reset: {retry_err}")
                        raise
                else:
                    # Some other ValueError we don't know how to handle
                    raise
            
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
                        
                print(f"Metrics - Train loss: {final_train_loss:.4f}, Validation loss: {final_valid_loss:.4f}")
                print(f"Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%), Error rate: {final_error_rate:.4f}")
                print(f"Learning rate: {learn.opt.hypers[-1]['lr'] if hasattr(learn, 'opt') and hasattr(learn.opt, 'hypers') else 'unknown'}")
                
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
            return
        
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
                'data_augmentation': augmentation,
                'date_created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_classes': len(dls.vocab),
                'class_names': list(dls.vocab),  # Convert CategoryMap to list
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
                    'class_names': list(dls.vocab),  # Convert CategoryMap to list
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
            
            # Process metrics if present to ensure safe JSON serialization
            if 'metrics' in status_update and isinstance(status_update['metrics'], dict):
                status_update['metrics'] = safe_metrics(status_update['metrics'])
                
            # Update status with new values
            status.update(status_update)
            
            # Convert to JSON-serializable format
            safe_status = {k: safe_json_value(v) for k, v in status.items()}
            
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
            args=(dataset_name, model_name, architecture, epochs, 
                  learning_rate, batch_size, 0.2, True, 224, data_augmentation)
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
    """Make status dictionary safe for JSON serialization."""
    if not status_dict:
        return {}
        
    try:
        # First try direct JSON conversion as a test
        json.dumps(status_dict)
        return status_dict
    except (TypeError, OverflowError):
        # If direct conversion fails, make a deep copy that is serializable
        
        def make_serializable(obj):
            """Recursively make an object JSON serializable."""
            if obj is None or isinstance(obj, (str, int, bool)):
                return obj
            elif isinstance(obj, (list, tuple, set)):
                try:
                    return [make_serializable(item) for item in obj]
                except:
                    try:
                        return list(obj)
                    except:
                        try:
                            return dict(obj)
                        except:
                            return str(obj)
            elif isinstance(obj, dict):
                # Process nested dictionaries
                return {str(k): make_serializable(v) for k, v in list(obj.items())}
            elif isinstance(obj, float):
                # Handle NaN, Infinity which are not JSON serializable
                if not isinstance(obj, bool) and not (float('-inf') < obj < float('inf')):
                    return str(obj)
                return obj
            elif hasattr(obj, '__dict__'):
                # Try to convert objects to dictionaries
                try:
                    return make_serializable(obj.__dict__)
                except:
                    pass
            
            # For fastai objects like CategoryMap
            if hasattr(obj, 'items'):
                try:
                    return list(obj.items)
                except:
                    pass
            
            # For other objects with item attribute
            if hasattr(obj, 'item'):
                try:
                    return obj.item()
                except:
                    pass
                
            # Last resort - convert to string
            return str(obj)
            
        # Apply recursive conversion to the entire status dictionary
        return make_serializable(status_dict)
        
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

def count_dataset_images(dataset_path):
    """
    Count the total number of images in a dataset.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        int: Total number of images across all classes
    """
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset path does not exist: {dataset_path}")
        return 0
        
    total_images = 0
    
    # Count images in each class directory
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            # Count image files with supported extensions
            image_count = sum(
                1 for f in os.listdir(class_dir) 
                if os.path.isfile(os.path.join(class_dir, f)) and 
                f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
            )
            total_images += image_count
            logger.debug(f"Class '{class_name}': {image_count} images")
    
    logger.info(f"Dataset contains a total of {total_images} images")
    return total_images 