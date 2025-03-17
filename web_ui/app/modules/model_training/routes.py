"""
Routes for the model training module.
"""
import os
import time
import logging
import traceback
from flask import (
    Blueprint, render_template, request, jsonify, 
    redirect, url_for, send_from_directory
)
from werkzeug.utils import secure_filename
import json

# Import modular services
from .utils import get_dataset_path, get_model_path, allowed_file
from .dataset_service import (
    get_datasets, get_dataset_stats, 
    create_dataset as create_dataset_service, 
    delete_dataset, create_example_dataset
)
from .training_service import (
    start_model_training, get_training_status, get_models
)

# Initialize logger
logger = logging.getLogger(__name__)

# Cache variables (for dataset info)
_dataset_info_cache = {}
_class_image_cache = {}
_cache_timeout = 300  # 5 minutes

# Definicja blueprintu
model_training_bp = Blueprint(
    'model_training', 
    __name__, 
    url_prefix='/train',
    template_folder='templates'
)


@model_training_bp.route('/')
def index():
    """Render the model training index page."""
    try:
        logger.info("Rendering model_training index page")
        return render_template('model_training/train_index.html')
    except Exception as e:
        logger.error(f"Error rendering index template: {e}")
        logger.error(traceback.format_exc())
        return str(e), 500


@model_training_bp.route('/new_model', methods=['GET', 'POST'])
def new_model():
    """Render the new model page."""
    try:
        # Get available datasets
        datasets = get_datasets()
        
        # Handle datasets format - could be list of dicts or list of strings
        if datasets and isinstance(datasets[0], dict):
            # Keep the original format for templates that expect it
            pass
        else:
            # Convert strings to dict format for backwards compatibility
            datasets = [{'name': d} for d in datasets]
        
        # Define available model architectures
        model_architectures = [
            "ResNet50", "MobileNetV2", "EfficientNetB0", 
            "VGG16", "InceptionV3", "DenseNet121"
        ]
        
        logger.info("Rendering new model template")
        return render_template(
            'model_training/new_model.html',
            datasets=datasets,
            model_architectures=model_architectures
        )
    except Exception as e:
        logger.error(f"Error rendering new model template: {e}")
        logger.error(traceback.format_exc())
        return str(e), 500


@model_training_bp.route('/finetune', methods=['GET', 'POST'])
def finetune():
    """Render the finetune model page."""
    try:
        # Get available datasets
        datasets = get_datasets()
        
        # Handle datasets format - could be list of dicts or list of strings
        if datasets and isinstance(datasets[0], dict):
            # Keep the original format for templates that expect it
            pass
        else:
            # Convert strings to dict format for backwards compatibility
            datasets = [{'name': d} for d in datasets]
        
        # Get available pre-trained models
        models = get_models()
        
        # If no saved models, offer pre-trained options
        if not models:
            models = [
                "ResNet50-ImageNet", "MobileNetV2-ImageNet", 
                "EfficientNetB0-ImageNet", "VGG16-ImageNet", 
                "InceptionV3-ImageNet"
            ]
        
        logger.info("Rendering finetune model template")
        return render_template(
            'model_training/finetune.html', 
            datasets=datasets,
            models=models
        )
    except Exception as e:
        logger.error(f"Error rendering finetune model template: {e}")
        logger.error(traceback.format_exc())
        return str(e), 500


@model_training_bp.route('/datasets')
def datasets():
    """Render the datasets page."""
    logger.info("Rendering datasets page")
    
    # Use cached data if available and not expired
    cache_key = 'all_datasets'
    if cache_key in _dataset_info_cache:
        cache_time, dataset_list = _dataset_info_cache[cache_key]
        if time.time() - cache_time < _cache_timeout:
            try:
                logger.debug(
                    f"Using cached dataset list with {len(dataset_list)} items"
                )
                
                # Handle both formats - list of dictionaries or list of strings
                if dataset_list and isinstance(dataset_list[0], dict):
                    datasets_names = [d['name'] for d in dataset_list]
                else:
                    # If cached list contains strings directly
                    datasets_names = dataset_list
                
                return render_template(
                    'model_training/datasets.html', 
                    datasets=datasets_names
                )
            except Exception as e:
                logger.error(f"Error rendering datasets template from cache: {e}")
                logger.error(traceback.format_exc())
                return str(e), 500
    
    # If not in cache or expired, get fresh data
    try:
        datasets_list = get_datasets()
        
        # Cache the results
        _dataset_info_cache[cache_key] = (time.time(), datasets_list)
        
        # Handle both formats - list of dictionaries or list of strings
        if datasets_list and isinstance(datasets_list[0], dict):
            datasets_names = [d['name'] for d in datasets_list]
        else:
            # If get_datasets returns a list of strings directly
            datasets_names = datasets_list
        
        return render_template(
            'model_training/datasets.html', 
            datasets=datasets_names
        )
    except Exception as e:
        logger.error(f"Error processing datasets: {e}")
        logger.error(traceback.format_exc())
        return str(e), 500


@model_training_bp.route('/datasets/new', methods=['GET', 'POST'])
def create_dataset():
    """Render the create dataset page or handle dataset creation."""
    try:
        # Handle dataset creation
        if request.method == 'POST':
            dataset_name = request.form.get('dataset_name')
            num_classes = int(request.form.get('num_classes', 2))
            class_names = [
                request.form.get(f'class_{i}', f'class_{i}') 
                for i in range(1, num_classes+1)
            ]
            
            # Create the dataset
            create_dataset_service(dataset_name, num_classes, class_names)
            
            return jsonify({
                'success': True,
                'dataset_name': dataset_name,
                'num_classes': num_classes,
                'redirect': url_for(
                    'model_training.edit_dataset', 
                    dataset_name=dataset_name
                )
            })
        
        # Render the create dataset form
        return render_template('model_training/create_dataset.html')
        
    except Exception as e:
        logger.error(f"Error handling create_dataset: {e}")
        logger.error(traceback.format_exc())
        if request.method == 'POST':
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
        return str(e), 500


@model_training_bp.route('/datasets/<dataset_name>/edit')
def edit_dataset(dataset_name):
    """Edit a dataset."""
    # Using logger instead of print
    logger.info(f"Attempting to edit dataset: {dataset_name}")
    
    try:
        # Get dataset path
        dataset_path = os.path.join(get_dataset_path(), dataset_name)
        logger.debug(f"Dataset path: {dataset_path}")
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset does not exist: {dataset_path}")
            return redirect(url_for('model_training.datasets'))
        
        # Collect class information
        classes = []
        
        # Scan classes in the dataset
        # Ensure the directory exists and is accessible
        if not os.path.isdir(dataset_path):
            logger.error(f"Dataset path is not a directory: {dataset_path}")
            return render_template(
                'error.html', 
                error=f"Dataset path is not a directory: {dataset_name}", 
                code=500
            ), 500
        
        try:
            class_list = os.listdir(dataset_path)
        except (PermissionError, OSError) as e:
            logger.error(f"Cannot list directory {dataset_path}: {str(e)}")
            return render_template(
                'error.html',
                error=f"Cannot access dataset: {str(e)}",
                code=500
            ), 500
        
        for class_name in class_list:
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                # Count images
                image_count = 0
                
                try:
                    files_list = os.listdir(class_path)
                except (PermissionError, OSError) as e:
                    logger.error(f"Cannot list files in class directory {class_path}: {str(e)}")
                    # Continue with next class instead of failing completely
                    continue
                
                for f in files_list:
                    file_path = os.path.join(class_path, f)
                    # Avoid errors with allowed_file by checking if the file exists first
                    if os.path.isfile(file_path) and allowed_file(f):
                        image_count += 1
                
                classes.append({
                    'name': class_name,
                    'count': image_count
                })
        
        logger.info(f"Found {len(classes)} classes")
        
        # Render template
        return render_template(
            'model_training/edit_dataset.html', 
            dataset_name=dataset_name, 
            classes=classes
        )
    
    except Exception as e:
        # Detailed error logging
        logger.error(f"Error in edit_dataset: {str(e)}")
        logger.error(traceback.format_exc())
        # Return a more user-friendly error page
        return render_template(
            'error.html',
            error=f"An error occurred while editing dataset: {str(e)}",
            code=500
        ), 500


@model_training_bp.route('/datasets/<dataset_name>/classes/new', methods=['POST'])
def new_class(dataset_name):
    """Add a new class to a dataset."""
    logger.info(f"Adding new class to dataset: {dataset_name}")
    try:
        class_name = request.form.get('class_name')
        
        if not class_name:
            logger.warning("Class name is required but not provided")
            return jsonify({
                'success': False,
                'error': 'Class name is required'
            }), 400
            
        # Create class directory
        class_path = os.path.join(
            get_dataset_path(), 
            dataset_name, 
            class_name
        )
        
        if os.path.exists(class_path):
            logger.warning(f"Class already exists: {class_name}")
            return jsonify({
                'success': False,
                'error': 'Class already exists'
            }), 400
            
        os.makedirs(class_path, exist_ok=True)
        
        # Clear cache for this dataset
        cache_key = f'dataset_{dataset_name}'
        if cache_key in _dataset_info_cache:
            del _dataset_info_cache[cache_key]
        
        logger.info(f"Class created successfully: {class_name} in dataset {dataset_name}")
        return jsonify({
            'success': True,
            'class_name': class_name
        })
    except Exception as e:
        logger.error(f"Error creating class: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Error creating class: {str(e)}'
        }), 500


@model_training_bp.route('/datasets/<dataset_name>/class/<class_name>/delete', methods=['POST'])
def delete_class(dataset_name, class_name):
    """Delete a class from a dataset."""
    logger.info(f"Deleting class {class_name} from dataset {dataset_name}")
    try:
        class_path = os.path.join(
            get_dataset_path(), 
            dataset_name, 
            class_name
        )
        
        if not os.path.exists(class_path):
            logger.warning(f"Class not found: {class_name}")
            return jsonify({
                'success': False,
                'error': 'Class not found'
            }), 404
            
        # Delete all files in the class directory
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        # Delete the class directory
        os.rmdir(class_path)
        
        # Clear cache for this dataset
        cache_key = f'dataset_{dataset_name}'
        if cache_key in _dataset_info_cache:
            del _dataset_info_cache[cache_key]
        
        # Clear image cache for this class
        cache_key = f'class_{dataset_name}_{class_name}'
        if cache_key in _class_image_cache:
            del _class_image_cache[cache_key]
        
        logger.info(f"Class deleted successfully: {class_name}")
        return jsonify({
            'success': True
        })
    except Exception as e:
        logger.error(f"Error deleting class: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@model_training_bp.route('/datasets/<dataset_name>/upload/<class_name>', methods=['POST'])
def upload_to_class(dataset_name, class_name):
    """Upload images to a specific class in a dataset."""
    logger.info(f"Uploading files to class {class_name} in dataset {dataset_name}")
    try:
        logger.debug(f"Request method: {request.method}")
        logger.debug(f"Request content type: {request.content_type}")
        logger.debug(f"Request files: {request.files.keys()}")
        
        if 'files[]' not in request.files:
            logger.warning("No files in request")
            return jsonify({
                'success': False,
                'error': 'No files uploaded'
            }), 400
            
        files = request.files.getlist('files[]')
        logger.debug(f"Number of files received: {len(files)}")
        
        if not files or files[0].filename == '':
            logger.warning("No files selected or empty filenames")
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
            
        # Get the class directory
        class_dir = os.path.join(
            get_dataset_path(), 
            dataset_name, 
            class_name
        )
        logger.debug(f"Class directory: {class_dir}")
        
        if not os.path.exists(class_dir):
            logger.warning(f"Class directory not found: {class_dir}")
            try:
                os.makedirs(class_dir, exist_ok=True)
                logger.info(f"Created class directory: {class_dir}")
            except Exception as e:
                logger.error(f"Error creating class directory: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Failed to create class directory: {str(e)}'
                }), 500
            
        # Use a list to track uploaded files and a counter for batch size
        saved_files = []
        processing_count = 0
        failed_files = []
        
        # Process files - simplify by using direct save
        for file in files:
            try:
                if file and file.filename:
                    logger.debug(f"Processing file: {file.filename}")
                    
                    if allowed_file(file.filename):
                        try:
                            # Generate a filename and save directly
                            filename = secure_filename(file.filename)
                            base, ext = os.path.splitext(filename)
                            unique_filename = f"{base}_{int(time.time() * 1000)}{ext}"
                            file_path = os.path.join(class_dir, unique_filename)
                            
                            # Save file directly
                            file.save(file_path)
                            
                            # Record success
                            saved_files.append(unique_filename)
                            processing_count += 1
                            logger.debug(f"Saved file: {unique_filename}")
                        except Exception as e:
                            # Log the error but continue processing other files
                            logger.error(f"Error saving file {file.filename}: {str(e)}")
                            failed_files.append({
                                'name': file.filename,
                                'error': str(e)
                            })
                    else:
                        logger.warning(f"Invalid file type: {file.filename}")
                        failed_files.append({
                            'name': file.filename,
                            'error': 'Invalid file type'
                        })
                else:
                    logger.warning("Empty file object or filename")
            except Exception as e:
                logger.error(f"Unexpected error processing file: {str(e)}")
                failed_files.append({
                    'name': getattr(file, 'filename', 'unknown'),
                    'error': str(e)
                })
        
        # Clear cache for this dataset and class
        cache_key = f'dataset_{dataset_name}'
        if cache_key in _dataset_info_cache:
            del _dataset_info_cache[cache_key]
            
        cache_key = f'class_{dataset_name}_{class_name}'
        if cache_key in _class_image_cache:
            del _class_image_cache[cache_key]
        
        if processing_count > 0:
            logger.info(f"Successfully processed {processing_count} files for class {class_name}")
            return jsonify({
                'success': True,
                'message': f'Successfully processed {processing_count} files for {class_name}',
                'files': saved_files,
                'failed': failed_files
            })
        else:
            logger.warning("No valid files were uploaded")
            return jsonify({
                'success': False,
                'error': 'No valid files were uploaded',
                'failed': failed_files
            }), 400
    except Exception as e:
        logger.error(f"Error processing uploads: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@model_training_bp.route('/datasets/<dataset_name>/stats', methods=['GET'])
def dataset_stats(dataset_name):
    """Get statistics about a dataset, including class counts."""
    logger.info(f"Getting stats for dataset: {dataset_name}")
    try:
        dataset_info = get_dataset_stats(dataset_name)
        
        if not dataset_info:
            return jsonify({
                'success': False,
                'error': f'Dataset {dataset_name} not found'
            }), 404
            
        return jsonify({
            'success': True,
            'dataset_name': dataset_name,
            'stats': dataset_info
        })
    except Exception as e:
        logger.error(f"Error getting dataset stats: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@model_training_bp.route('/datasets/<dataset_name>/class/<class_name>/images', methods=['GET'])
def get_class_images(dataset_name, class_name):
    """Get all images for a specific class in a dataset."""
    logger.info(f"Getting images for class {class_name} in dataset {dataset_name}")
    try:
        class_dir = os.path.join(get_dataset_path(), dataset_name, class_name)
        
        if not os.path.exists(class_dir):
            logger.warning(f"Class directory not found: {class_dir}")
            return jsonify({
                'success': False,
                'error': 'Class not found'
            }), 404
            
        # Check cache first
        cache_key = f'class_{dataset_name}_{class_name}'
        cache_entry = _class_image_cache.get(cache_key)
        if cache_entry:
            cache_time, images = cache_entry
            if time.time() - cache_time < _cache_timeout:
                return jsonify({
                    'success': True,
                    'dataset_name': dataset_name,
                    'class_name': class_name,
                    'images': images
                })
                
        # Get all images in the class directory
        images = []
        for filename in os.listdir(class_dir):
            if os.path.isfile(os.path.join(class_dir, filename)) and allowed_file(filename):
                # Create a URL for the image using our dataset_files route
                image_url = url_for('model_training.dataset_files', 
                                   filename=f'{dataset_name}/{class_name}/{filename}')
                
                images.append({
                    'filename': filename,
                    'url': image_url
                })
                
        # Cache the result
        _class_image_cache[cache_key] = (time.time(), images)
        
        return jsonify({
            'success': True,
            'dataset_name': dataset_name,
            'class_name': class_name,
            'images': images
        })
    except Exception as e:
        logger.error(f"Error getting class images: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@model_training_bp.route('/datasets/files/<path:filename>')
def dataset_files(filename):
    """Serve files from the dataset directory."""
    logger.debug(f"Serving dataset file: {filename}")
    return send_from_directory(get_dataset_path(), filename)


@model_training_bp.route('/datasets/example', methods=['GET', 'POST'])
def example_datasets():
    """Render the example datasets page or handle dataset downloads."""
    try:
        # Handle dataset download requests
        if request.method == 'POST':
            dataset_name = request.form.get('dataset_name')
            if not dataset_name:
                return jsonify({
                    'success': False,
                    'error': 'Dataset name is required'
                }), 400
                
            # Create example dataset with real images
            logger.info(f"Creating example dataset with images: {dataset_name}")
            
            # Create the example dataset
            created_dataset_name, classes = create_example_dataset(dataset_name, dataset_name)
                
            logger.info(f"Started downloading images for dataset: {created_dataset_name}")
            
            # Create a response for the client indicating success
            return jsonify({
                'success': True,
                'message': f'Dataset {created_dataset_name} created and downloading images',
                'redirect': url_for('model_training.edit_dataset', dataset_name=created_dataset_name)
            })
        
        # Handle GET requests - show the example datasets page
        logger.info("Rendering example datasets page")
        # Sample data for example datasets
        datasets = {
            "Pet Animals": "/static/example_datasets/pets.zip",
            "Flowers": "/static/example_datasets/flowers.zip",
            "Food Items": "/static/example_datasets/food.zip"
        }
        return render_template('model_training/example_datasets.html', datasets=datasets)
    except Exception as e:
        logger.error(f"Error handling example_datasets: {e}")
        logger.error(traceback.format_exc())
        if request.method == 'POST':
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
        return str(e), 500


@model_training_bp.route('/datasets/webscrape')
def webscrape():
    """Render the webscrape page."""
    logger.info("Rendering webscrape page")
    try:
        return render_template('model_training/webscrape.html')
    except Exception as e:
        logger.error(f"Error rendering webscrape template: {e}")
        logger.error(traceback.format_exc())
        return str(e), 500


@model_training_bp.route('/tutorial')
def ml_tutorial():
    """Render the machine learning tutorial page."""
    try:
        logger.info("Rendering ML tutorial page")
        return render_template('model_training/ml_tutorial.html')
    except Exception as e:
        logger.error(f"Error rendering ML tutorial template: {e}")
        logger.error(traceback.format_exc())
        return str(e), 500


@model_training_bp.route('/datasets/<dataset_name>/delete', methods=['POST'])
def delete_dataset_route(dataset_name):
    """Delete a dataset."""
    logger.info(f"Deleting dataset: {dataset_name}")
    try:
        success = delete_dataset(dataset_name)
        
        if not success:
            return jsonify({
                'success': False,
                'error': 'Failed to delete dataset'
            }), 500
            
        logger.info(f"Dataset deleted successfully: {dataset_name}")
        return jsonify({
            'success': True,
            'redirect': url_for('model_training.datasets')
        })
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@model_training_bp.route('/models/train', methods=['POST'])
def train_model():
    """Start training a model with the specified parameters."""
    try:
        logger.info("Received request to train model")
        
        # Get form data
        model_name = request.form.get('model_name', '')
        dataset_name = request.form.get('dataset', '')
        architecture = request.form.get('architecture', '')
        
        logger.debug(f"Received training request for model: {model_name}, dataset: {dataset_name}")
        
        if not model_name:
            logger.error("Missing model_name field")
            return jsonify({'success': False, 'error': 'Model name is required'}), 400
            
        if not dataset_name:
            logger.error("Missing dataset field")
            return jsonify({'success': False, 'error': 'Dataset is required'}), 400
            
        # Extract dataset name if it's in JSON format (coming from form)
        try:
            import json
            dataset_dict = json.loads(dataset_name)
            if isinstance(dataset_dict, dict) and 'name' in dataset_dict:
                dataset_name = dataset_dict['name']
                logger.info(f"Extracted dataset name from JSON: {dataset_name}")
        except (json.JSONDecodeError, TypeError):
            # Not JSON format, keep as is
            pass
            
        if not architecture:
            logger.error("Missing architecture field")
            return jsonify({'success': False, 'error': 'Architecture is required'}), 400
        
        # Get optional fields with defaults
        try:
            epochs = int(request.form.get('epochs', 5))
        except (ValueError, TypeError):
            epochs = 5
            
        try:
            batch_size = int(request.form.get('batch_size', 8))
        except (ValueError, TypeError):
            batch_size = 8
            
        try:
            learning_rate = float(request.form.get('learning_rate', 0.001))
        except (ValueError, TypeError):
            learning_rate = 0.001
            
        data_augmentation = request.form.get('data_augmentation', 'off') == 'on'
        
        # Validate dataset exists
        dataset_path = os.path.join(get_dataset_path(), dataset_name)
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset {dataset_name} not found")
            return jsonify({
                'success': False, 
                'error': f'Dataset {dataset_name} not found'
            }), 404
            
        # Start training in background thread
        success = start_model_training(
            model_name, 
            dataset_name,
            architecture,
            epochs,
            batch_size,
            learning_rate,
            data_augmentation
        )
        
        if success:
            # Return immediate success response
            logger.info(f"Model training started for {model_name}")
            return jsonify({
                'success': True,
                'message': f'Training started for model: {model_name}',
                'redirect': url_for('model_training.trainings')
            })
        else:
            logger.error(f"Failed to start training for {model_name}")
            return jsonify({
                'success': False,
                'error': 'Failed to start training process'
            }), 500
            
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@model_training_bp.route('/trainings')
def trainings():
    """Render the trainings view page."""
    try:
        logger.info("Rendering trainings view page")
        return render_template('model_training/train_index.html', active_tab='trainings')
    except Exception as e:
        logger.error(f"Error rendering trainings template: {e}")
        logger.error(traceback.format_exc())
        return str(e), 500


@model_training_bp.route('/trainings/status')
def trainings_status():
    """Get status of all training tasks."""
    try:
        # Get directory containing training status files
        status_dir = os.path.join(get_model_path(), 'training_status')
        
        if not os.path.exists(status_dir):
            return jsonify({
                'active': [],
                'completed': [],
                'failed': []
            })
            
        # Get all status files
        status_files = [f for f in os.listdir(status_dir) if f.endswith('.json')]
        
        # Categorize trainings
        active_trainings = []
        completed_trainings = []
        failed_trainings = []
        
        # Current time for calculating duration
        current_time = time.time()
        
        for status_file in status_files:
            try:
                # Get model name from filename
                model_name = os.path.splitext(status_file)[0]
                
                # Read status file
                with open(os.path.join(status_dir, status_file), 'r') as f:
                    status = json.load(f)
                    
                # Add model name if not present
                if 'model_name' not in status:
                    status['model_name'] = model_name
                    
                # Calculate duration if possible
                if 'started_at' in status:
                    status['duration'] = int(current_time - float(status['started_at']))
                
                # Categorize based on status
                if status.get('status') == 'completed':
                    completed_trainings.append(status)
                elif status.get('status') == 'error':
                    failed_trainings.append(status)
                else:
                    # Check if active training is stale (no updates in 10 minutes)
                    is_stale = False
                    status_file_path = os.path.join(status_dir, status_file)
                    last_modified = os.path.getmtime(status_file_path)
                    
                    if current_time - last_modified > 600:  # 10 minutes
                        # Mark as stale and move to failed
                        status['status'] = 'error'
                        status['error'] = 'Training process appears to be stalled'
                        status['error_type'] = 'stalled'
                        failed_trainings.append(status)
                    else:
                        active_trainings.append(status)
            except Exception as e:
                logger.error(f"Error processing training status file {status_file}: {e}")
                # Add to failed with error info
                failed_trainings.append({
                    'model_name': os.path.splitext(status_file)[0],
                    'status': 'error',
                    'error': f"Error reading status: {str(e)}"
                })
                
        # Sort trainings: active by start time (recent first), completed by completion time
        active_trainings.sort(key=lambda x: x.get('started_at', 0), reverse=True)
        completed_trainings.sort(key=lambda x: x.get('completed_at', 0), reverse=True)
        
        return jsonify({
            'active': active_trainings,
            'completed': completed_trainings,
            'failed': failed_trainings
        })
    except Exception as e:
        logger.error(f"Error getting training statuses: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@model_training_bp.route('/models/<model_name>/status')
def model_status(model_name):
    """Get the current status of a model's training."""
    try:
        status = get_training_status(model_name)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500 