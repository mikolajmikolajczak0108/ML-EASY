"""
Utility functions for file operations.
"""
import os
import zipfile
import io
import requests
from flask import current_app
from werkzeug.utils import secure_filename


def allowed_file(filename):
    """
    Check if a file has an allowed extension.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in 
            current_app.config['ALLOWED_EXTENSIONS'])


def is_video_file(filename):
    """
    Check if a file is a video.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        bool: True if file is a video, False otherwise
    """
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in {'mp4', 'mov'})


def save_uploaded_file(file, directory=None):
    """
    Save an uploaded file to the specified directory.
    
    Args:
        file: File object from request.files
        directory: Specific subdirectory to save to (optional)
        
    Returns:
        str: Path to the saved file
    """
    filename = secure_filename(file.filename)
    
    # Generate a unique filename to prevent overwriting
    base, ext = os.path.splitext(filename)
    unique_filename = f"{base}_{os.urandom(4).hex()}{ext}"
    
    # Set the save directory
    upload_folder = current_app.config['UPLOAD_FOLDER']
    if directory:
        save_dir = os.path.join(upload_folder, directory)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = upload_folder
        
    file_path = os.path.join(save_dir, unique_filename)
    file.save(file_path)
    
    return file_path


def download_and_extract_dataset(dataset_name, dataset_url):
    """
    Download and extract a dataset from a URL.
    
    Args:
        dataset_name: Name of the dataset
        dataset_url: URL to download the dataset from
        
    Returns:
        bool: True if successful, False otherwise
    """
    dataset_dir = os.path.join(
        current_app.config['DATASET_PATH'], dataset_name)
    
    if os.path.exists(dataset_dir):
        return True
    
    try:
        os.makedirs(dataset_dir, exist_ok=True)
        
        response = requests.get(dataset_url, stream=True)
        if response.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(dataset_dir)
            return True
        else:
            return False
    except Exception:
        return False 