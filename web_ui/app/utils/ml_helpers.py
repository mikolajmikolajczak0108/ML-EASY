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
        list: List of model filenames
    """
    models_path = current_app.config['MODEL_PATH']
    return [f for f in os.listdir(models_path) if f.endswith('.pkl')]


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
    Load a trained fastai model.
    
    Args:
        model_name: Name of the model file
        
    Returns:
        model: Loaded model or None if loading fails
    """
    model_path = os.path.join(current_app.config['MODEL_PATH'], model_name)
    try:
        if os.path.exists(model_path):
            return load_learner(model_path)
        return None
    except Exception:
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