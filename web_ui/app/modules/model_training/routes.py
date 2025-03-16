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

# Konfiguracja loggera
logger = logging.getLogger(__name__)

# Cache dla danych
_dataset_info_cache = {}
_class_image_cache = {}
_cache_timeout = 60  # Cache timeout in seconds (refresh every minute)

# Lista dozwolonych rozszerzeń - stała zamiast pobierania z config
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov'}

# Ścieżki do katalogów (będą uzupełniane przy pierwszym użyciu)
_BASE_PATH = None
_DATASET_PATH = None
_MODEL_PATH = None
_UPLOAD_PATH = None

def get_dataset_path():
    """Get the dataset path, ensuring it exists within an application context."""
    global _DATASET_PATH, _BASE_PATH
    if _DATASET_PATH is None:
        _BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        _DATASET_PATH = os.path.join(_BASE_PATH, 'datasets')
        os.makedirs(_DATASET_PATH, exist_ok=True)
    return _DATASET_PATH

def get_model_path():
    """Get the model path, ensuring it exists within an application context."""
    global _MODEL_PATH, _BASE_PATH
    if _MODEL_PATH is None:
        if _BASE_PATH is None:
            _BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
        _MODEL_PATH = os.path.join(_BASE_PATH, 'models')
        os.makedirs(_MODEL_PATH, exist_ok=True)
    return _MODEL_PATH

def get_upload_path():
    """Get the upload path, ensuring it exists within an application context."""
    global _UPLOAD_PATH, _BASE_PATH
    if _UPLOAD_PATH is None:
        if _BASE_PATH is None:
            _BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
        _UPLOAD_PATH = os.path.join(_BASE_PATH, 'uploads')
        os.makedirs(_UPLOAD_PATH, exist_ok=True)
    return _UPLOAD_PATH

# Funkcja do sprawdzania rozszerzeń plików 
def allowed_file(filename):
    """Check if a file has an allowed extension."""
    if not filename or not isinstance(filename, str):
        return False
    
    # Make sure filename has an extension
    if '.' not in filename:
        return False
    
    # Get the last part after the dot as the extension
    try:
        ext = filename.rsplit('.', 1)[1].lower()
        return ext in ALLOWED_EXTENSIONS
    except (IndexError, AttributeError):
        return False

# Definicja blueprintu
model_training_bp = Blueprint(
    'model_training', 
    __name__, 
    url_prefix='/train',
    template_folder='templates/model_training'
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


@model_training_bp.route('/new-model')
def new_model():
    """Render the new model page."""
    try:
        logger.info("Rendering new_model page")
        return render_template('model_training/new_model.html')
    except Exception as e:
        logger.error(f"Error rendering new_model template: {e}")
        logger.error(traceback.format_exc())
        return str(e), 500


@model_training_bp.route('/finetune')
def finetune():
    """Render the finetune page."""
    try:
        logger.info("Rendering finetune page")
        return render_template('model_training/finetune.html')
    except Exception as e:
        logger.error(f"Error rendering finetune template: {e}")
        logger.error(traceback.format_exc())
        return str(e), 500


@model_training_bp.route('/datasets')
def datasets():
    """Render the datasets page."""
    logger.info("Rendering datasets page")
    # Get all datasets
    datasets_path = get_dataset_path()
    
    # Use cached data if available and not expired
    cache_key = 'all_datasets'
    if cache_key in _dataset_info_cache:
        cache_time, dataset_list = _dataset_info_cache[cache_key]
        if time.time() - cache_time < _cache_timeout:
            try:
                logger.debug(f"Using cached dataset list with {len(dataset_list)} items")
                return render_template(
                    'model_training/datasets.html', 
                    datasets=[d['name'] for d in dataset_list]
                )
            except Exception as e:
                logger.error(f"Error rendering datasets template from cache: {e}")
                logger.error(traceback.format_exc())
                return str(e), 500
    
    # If not in cache or expired, get fresh data
    dataset_list = []
    dataset_names = []
    
    try:
        if os.path.exists(datasets_path):
            logger.debug(f"Scanning dataset path: {datasets_path}")
            for dataset_name in os.listdir(datasets_path):
                dataset_path = os.path.join(datasets_path, dataset_name)
                if os.path.isdir(dataset_path):
                    # Get class count
                    class_count = 0
                    classes = []
                    for class_name in os.listdir(dataset_path):
                        class_path = os.path.join(dataset_path, class_name)
                        if os.path.isdir(class_path):
                            class_count += 1
                            classes.append(class_name)
                    
                    dataset_list.append({
                        'name': dataset_name,
                        'class_count': class_count,
                        'classes': classes
                    })
                    dataset_names.append(dataset_name)
            
            logger.debug(f"Found {len(dataset_list)} datasets")
        else:
            logger.warning(f"Dataset path doesn't exist: {datasets_path}")
        
        # Cache the results
        _dataset_info_cache[cache_key] = (time.time(), dataset_list)
        
        return render_template(
            'model_training/datasets.html', 
            datasets=dataset_names
        )
    except Exception as e:
        logger.error(f"Error processing datasets: {e}")
        logger.error(traceback.format_exc())
        return str(e), 500


@model_training_bp.route('/datasets/new', methods=['GET', 'POST'])
def create_dataset():
    """Create a new dataset."""
    if request.method == 'GET':
        logger.info("Rendering new_dataset page")
        try:
            return render_template('model_training/new_dataset.html')
        except Exception as e:
            logger.error(f"Error rendering new_dataset template: {e}")
            logger.error(traceback.format_exc())
            return str(e), 500
        
    # Process POST request to create a dataset
    try:
        logger.info("Processing new dataset creation")
        dataset_name = request.form.get('dataset_name')
        if not dataset_name:
            logger.warning("Dataset name is required but not provided")
            return jsonify({
                'success': False,
                'error': 'Dataset name is required'
            }), 400
            
        # Create dataset directory
        dataset_path = os.path.join(
            get_dataset_path(), dataset_name)
        if os.path.exists(dataset_path):
            logger.warning(f"Dataset already exists: {dataset_name}")
            return jsonify({
                'success': False,
                'error': 'Dataset with this name already exists'
            }), 400
            
        os.makedirs(dataset_path, exist_ok=True)
        
        # Get number of classes
        num_classes = int(request.form.get('num_classes', 2))
        class_names = request.form.getlist('class_names[]')
        
        # Create folders for each class
        for i in range(num_classes):
            class_name = class_names[i] if i < len(class_names) else f"class_{i+1}"
            os.makedirs(os.path.join(dataset_path, class_name), exist_ok=True)
            
        logger.info(f"Dataset created successfully: {dataset_name} with {num_classes} classes")
        return jsonify({
            'success': True,
            'dataset_name': dataset_name,
            'num_classes': num_classes,
            'redirect': url_for('model_training.edit_dataset', dataset_name=dataset_name)
        })
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Error creating dataset: {str(e)}'
        }), 500


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
        return render_template('model_training/edit_dataset.html', 
                             dataset_name=dataset_name, 
                             classes=classes)
    
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
        dataset_dir = os.path.join(get_dataset_path(), dataset_name)
        
        if not os.path.exists(dataset_dir):
            logger.warning(f"Dataset not found: {dataset_name}")
            return jsonify({
                'success': False,
                'error': 'Dataset not found'
            }), 404
            
        class_counts = {}
        total_images = 0
        
        # Get all classes in the dataset
        for class_name in os.listdir(dataset_dir):
            class_dir = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_dir):
                # Count images in this class
                image_count = 0
                for f in os.listdir(class_dir):
                    if os.path.isfile(os.path.join(class_dir, f)) and allowed_file(f):
                        image_count += 1
                
                class_counts[class_name] = image_count
                total_images += image_count
        
        logger.debug(f"Stats for dataset {dataset_name}: {len(class_counts)} classes, {total_images} images")
        return jsonify({
            'success': True,
            'dataset_name': dataset_name,
            'total_images': total_images,
            'class_count': len(class_counts),
            'class_counts': class_counts
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
            
        images = []
        
        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            if os.path.isfile(file_path) and allowed_file(filename):
                # Create a URL for the image using our dataset_files route
                image_url = url_for('model_training.dataset_files', 
                                   filename=f'{dataset_name}/{class_name}/{filename}')
                
                images.append({
                    'filename': filename,
                    'url': image_url
                })
        
        logger.debug(f"Found {len(images)} images for class {class_name}")
        return jsonify({
            'success': True,
            'class_name': class_name,
            'dataset_name': dataset_name,
            'image_count': len(images),
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


@model_training_bp.route('/datasets/example')
def example_datasets():
    """Render the example datasets page."""
    logger.info("Rendering example datasets page")
    try:
        # Sample data for example datasets
        datasets = {
            "Pet Animals": "/static/example_datasets/pets.zip",
            "Flowers": "/static/example_datasets/flowers.zip",
            "Food Items": "/static/example_datasets/food.zip"
        }
        return render_template('model_training/example_datasets.html', datasets=datasets)
    except Exception as e:
        logger.error(f"Error rendering example_datasets template: {e}")
        logger.error(traceback.format_exc())
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