"""
Utility functions for file operations.
"""
import os
import zipfile
import io
import requests
from flask import current_app
from werkzeug.utils import secure_filename
import time
import threading
import queue


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
    # Using timestamp is faster than os.urandom
    base, ext = os.path.splitext(filename)
    unique_filename = f"{base}_{int(time.time() * 1000)}{ext}"
    
    # Set the save directory
    upload_folder = current_app.config['UPLOAD_FOLDER']
    if directory:
        # Handle special case for dataset uploads
        if directory.startswith("datasets/"):
            save_dir = os.path.join(current_app.config['DATASET_PATH'], directory.replace("datasets/", ""))
        elif "/" in directory:
            # For dataset uploads directly specifying the path
            dataset_parts = directory.split("/")
            if len(dataset_parts) == 2:
                save_dir = os.path.join(current_app.config['DATASET_PATH'], dataset_parts[0], dataset_parts[1])
            else:
                save_dir = os.path.join(upload_folder, directory)
        else:
            save_dir = os.path.join(upload_folder, directory)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = upload_folder
        
    file_path = os.path.join(save_dir, unique_filename)
    
    # Save the file directly without loading it all into memory
    try:
        # Check if file is a BytesIO object or a FileStorage
        if hasattr(file, 'read') and callable(file.read):
            # For in-memory BytesIO files
            with open(file_path, 'wb') as f:
                f.write(file.read())
                
            # If it's a FileStorage, we need to seek back to beginning for reuse
            if hasattr(file, 'seek') and callable(file.seek):
                file.seek(0)
        else:
            # For standard Flask FileStorage objects
            file.save(file_path)
            
        return file_path
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        raise e


# Global queue for file processing
file_queue = queue.Queue()
# Flag to track if worker thread is running
worker_running = False


def process_file_queue():
    """Process files in the queue one by one."""
    global worker_running
    worker_running = True
    print("Starting file queue worker thread")
    
    try:
        while True:
            try:
                # Get an item with a 5-second timeout
                file, directory, callback = file_queue.get(timeout=5)
                
                try:
                    # Process the file
                    print(f"Processing file for directory: {directory}")
                    file_path = save_uploaded_file(file, directory)
                    print(f"Saved file to: {file_path}")
                    
                    # Call the callback with the result
                    if callback:
                        callback(True, file_path)
                except Exception as e:
                    print(f"Error processing file: {str(e)}")
                    # Call the callback with the error
                    if callback:
                        callback(False, str(e))
                finally:
                    file_queue.task_done()
            except queue.Empty:
                # If the queue has been empty for 5 seconds, exit the thread
                print("File queue empty, stopping worker thread")
                break
    except Exception as e:
        # Log the error but don't crash
        print(f"Error in file queue processing: {str(e)}")
    finally:
        # Reset the worker flag when done
        worker_running = False
        print("File queue worker thread stopped")


def queue_file_upload(file, directory=None, callback=None):
    """
    Queue a file for upload processing in the background.
    
    Args:
        file: File object from request.files
        directory: Specific subdirectory to save to (optional)
        callback: Function to call when file is saved (optional)
        
    Returns:
        None
    """
    global worker_running
    
    try:
        print(f"Queueing file upload for directory: {directory}")
        
        # For FileStorage objects, we need to make a copy to avoid issues when the request context ends
        if hasattr(file, 'stream') and hasattr(file, 'filename'):
            # Create a BytesIO copy
            file_copy = io.BytesIO()
            file.save(file_copy)
            file_copy.seek(0)  # Reset position to beginning
            
            # This ensures the file name is preserved
            file_copy.filename = file.filename
            file = file_copy
        
        # Put the file in the queue
        file_queue.put((file, directory, callback))
        
        # Start worker thread if not running
        if not worker_running:
            thread = threading.Thread(target=process_file_queue)
            thread.daemon = True
            thread.start()
            
    except Exception as e:
        # Log any errors during queueing
        print(f"Error queueing file upload: {str(e)}")
        # Call the callback with the error if provided
        if callback:
            callback(False, str(e))


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