"""
Routes for the model training module.
"""
import os
import json
from flask import (
    Blueprint, render_template, request, jsonify, 
    current_app, redirect, url_for
)

from app.utils.file_helpers import (
    allowed_file, download_and_extract_dataset, save_uploaded_file
)
from app.utils.ml_helpers import (
    AVAILABLE_ARCHITECTURES, EXAMPLE_DATASETS,
    get_available_datasets, create_datablock
)

# Create blueprint
model_training_bp = Blueprint(
    'model_training', 
    __name__, 
    url_prefix='/train',
    template_folder='../../templates/model_training'
)


@model_training_bp.route('/')
def index():
    """Render the model training homepage."""
    datasets = get_available_datasets()
    return render_template(
        'train_index.html', 
        datasets=datasets,
        architectures=list(AVAILABLE_ARCHITECTURES.keys())
    )


@model_training_bp.route('/new_model', methods=['GET', 'POST'])
def train_new_model():
    """Train a new model from scratch."""
    if request.method == 'GET':
        datasets = get_available_datasets()
        return render_template(
            'new_model.html', 
            datasets=datasets,
            architectures=list(AVAILABLE_ARCHITECTURES.keys())
        )
    
    # POST logic for training will be implemented in the next phase
    return jsonify({
        'success': False,
        'error': 'Training a new model not implemented yet'
    }), 501


@model_training_bp.route('/finetune', methods=['GET', 'POST'])
def finetune_model():
    """Finetune an existing model."""
    if request.method == 'GET':
        datasets = get_available_datasets()
        # TODO: Add list of existing models
        return render_template(
            'finetune.html', 
            datasets=datasets
        )
    
    # POST logic for finetuning will be implemented in the next phase
    return jsonify({
        'success': False,
        'error': 'Finetuning a model not implemented yet'
    }), 501


@model_training_bp.route('/datasets', methods=['GET'])
def list_datasets():
    """List all available datasets."""
    datasets = get_available_datasets()
    return render_template('datasets.html', datasets=datasets)


@model_training_bp.route('/datasets/new', methods=['GET', 'POST'])
def create_dataset():
    """Create a new dataset."""
    if request.method == 'GET':
        return render_template('new_dataset.html')
        
    # Process POST request to create a dataset
    try:
        dataset_name = request.form.get('dataset_name')
        if not dataset_name:
            return jsonify({
                'success': False,
                'error': 'Dataset name is required'
            }), 400
            
        # Create dataset directory
        dataset_path = os.path.join(
            current_app.config['DATASET_PATH'], dataset_name)
        if os.path.exists(dataset_path):
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
            
        return jsonify({
            'success': True,
            'dataset_name': dataset_name,
            'num_classes': num_classes,
            'redirect': url_for('model_training.edit_dataset', dataset_name=dataset_name)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error creating dataset: {str(e)}'
        }), 500


@model_training_bp.route('/datasets/<dataset_name>/edit', methods=['GET'])
def edit_dataset(dataset_name):
    """Edit a dataset by adding images to classes."""
    dataset_path = os.path.join(current_app.config['DATASET_PATH'], dataset_name)
    if not os.path.exists(dataset_path):
        return redirect(url_for('model_training.list_datasets'))
        
    classes = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d))]
              
    return render_template(
        'edit_dataset.html', 
        dataset_name=dataset_name, 
        classes=classes
    )


@model_training_bp.route('/datasets/<dataset_name>/upload/<class_name>', methods=['POST'])
def upload_to_class(dataset_name, class_name):
    """Upload images to a specific class in a dataset."""
    if 'files[]' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No files uploaded'
        }), 400
        
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({
            'success': False,
            'error': 'No files selected'
        }), 400
        
    # Get the class directory
    class_dir = os.path.join(
        current_app.config['DATASET_PATH'], 
        dataset_name, 
        class_name
    )
    
    if not os.path.exists(class_dir):
        return jsonify({
            'success': False,
            'error': 'Class not found'
        }), 404
        
    # Save each file to the class directory
    saved_files = []
    for file in files:
        if allowed_file(file.filename):
            file_path = save_uploaded_file(file, os.path.join(dataset_name, class_name))
            saved_files.append(os.path.basename(file_path))
            
    return jsonify({
        'success': True,
        'message': f'Uploaded {len(saved_files)} files to {class_name}',
        'files': saved_files
    })


@model_training_bp.route('/datasets/example', methods=['GET', 'POST'])
def example_datasets():
    """Handle example datasets."""
    if request.method == 'GET':
        return render_template(
            'example_datasets.html', 
            datasets=EXAMPLE_DATASETS
        )
        
    # Process POST request to download example dataset
    dataset_name = request.form.get('dataset_name')
    if not dataset_name or dataset_name not in EXAMPLE_DATASETS:
        return jsonify({
            'success': False,
            'error': 'Invalid dataset selected'
        }), 400
        
    dataset_url = EXAMPLE_DATASETS[dataset_name]
    success = download_and_extract_dataset(dataset_name, dataset_url)
    
    if success:
        return jsonify({
            'success': True,
            'message': f'Dataset {dataset_name} downloaded successfully',
            'redirect': url_for('model_training.list_datasets')
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to download dataset'
        }), 500


@model_training_bp.route('/datasets/webscrape', methods=['GET', 'POST'])
def webscrape_dataset():
    """Create a dataset by web scraping images."""
    if request.method == 'GET':
        return render_template('webscrape.html')
        
    # POST logic for web scraping will be implemented in the next phase
    return jsonify({
        'success': False,
        'error': 'Web scraping for datasets not implemented yet'
    }), 501 