"""
Integration tests for dataset routes.
"""
import os
import unittest
import tempfile
import shutil
from unittest.mock import patch
from io import BytesIO

from app import create_app


class TestDatasetRoutes(unittest.TestCase):
    """Test cases for the dataset routes."""
    
    def setUp(self):
        """Set up the test environment."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.app.config['DATASET_PATH'] = os.path.join(self.temp_dir, 'datasets')
        self.app.config['UPLOAD_FOLDER'] = os.path.join(self.temp_dir, 'uploads')
        
        # Ensure directories exist
        os.makedirs(self.app.config['DATASET_PATH'], exist_ok=True)
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def tearDown(self):
        """Tear down the test environment."""
        self.app_context.pop()
        # Remove temp directories
        shutil.rmtree(self.temp_dir)
    
    def test_datasets_page(self):
        """Test the datasets page route."""
        response = self.client.get('/train/datasets')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Datasets', response.data)
    
    def test_new_dataset_page(self):
        """Test the new dataset page route."""
        response = self.client.get('/train/datasets/new')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Create New Dataset', response.data)
    
    def test_create_dataset(self):
        """Test creating a new dataset."""
        # Test POST to create dataset
        data = {
            'dataset_name': 'test_dataset',
            'num_classes': 2,
            'class_names[]': ['class1', 'class2']
        }
        response = self.client.post('/train/datasets/new', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        
        # Verify response is JSON with success
        json_data = response.json
        self.assertTrue(json_data.get('success'))
        self.assertEqual(json_data.get('dataset_name'), 'test_dataset')
        
        # Verify directory was created
        dataset_path = os.path.join(self.app.config['DATASET_PATH'], 'test_dataset')
        self.assertTrue(os.path.exists(dataset_path))
        
        # Verify class directories were created
        self.assertTrue(os.path.exists(os.path.join(dataset_path, 'class1')))
        self.assertTrue(os.path.exists(os.path.join(dataset_path, 'class2')))
    
    def test_edit_dataset(self):
        """Test editing a dataset."""
        # Create test dataset first
        dataset_path = os.path.join(self.app.config['DATASET_PATH'], 'test_edit_dataset')
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_path, 'class1'), exist_ok=True)
        
        # Test GET to edit dataset
        response = self.client.get('/train/datasets/test_edit_dataset/edit')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'test_edit_dataset', response.data)
    
    def test_add_class_to_dataset(self):
        """Test adding a class to an existing dataset."""
        # Create test dataset first
        dataset_path = os.path.join(self.app.config['DATASET_PATH'], 'test_add_class')
        os.makedirs(dataset_path, exist_ok=True)
        
        # Test POST to add class
        data = {'class_name': 'new_class'}
        response = self.client.post(
            '/train/datasets/test_add_class/classes/new', 
            data=data
        )
        self.assertEqual(response.status_code, 200)
        
        # Verify response is JSON with success
        json_data = response.json
        self.assertTrue(json_data.get('success'))
        
        # Verify class directory was created
        self.assertTrue(os.path.exists(os.path.join(dataset_path, 'new_class')))
    
    def test_delete_class(self):
        """Test deleting a class from a dataset."""
        # Create test dataset and class first
        dataset_path = os.path.join(self.app.config['DATASET_PATH'], 'test_delete')
        class_path = os.path.join(dataset_path, 'class_to_delete')
        os.makedirs(class_path, exist_ok=True)
        
        # Test POST to delete class
        response = self.client.post(
            '/train/datasets/test_delete/class/class_to_delete/delete'
        )
        self.assertEqual(response.status_code, 200)
        
        # Verify response is JSON with success
        json_data = response.json
        self.assertTrue(json_data.get('success'))
        
        # Verify class directory was deleted
        self.assertFalse(os.path.exists(class_path))
    
    def test_upload_to_class(self):
        """Test uploading files to a class."""
        # Create test dataset and class first
        dataset_path = os.path.join(self.app.config['DATASET_PATH'], 'test_upload')
        class_path = os.path.join(dataset_path, 'test_class')
        os.makedirs(class_path, exist_ok=True)
        
        # Create a fake image file
        image_data = BytesIO(b'fake image data')
        
        # Test POST to upload file
        data = {'files[]': (image_data, 'test.jpg')}
        
        response = self.client.post(
            '/train/datasets/test_upload/upload/test_class',
            data=data,
            content_type='multipart/form-data'
        )
        
        self.assertEqual(response.status_code, 200)
        
        # Verify response is JSON with success
        json_data = response.json
        self.assertTrue(json_data.get('success'))
        
        # Verify at least one file was saved
        files = os.listdir(class_path)
        self.assertGreaterEqual(len(files), 1)
    
    def test_dataset_stats(self):
        """Test getting dataset statistics."""
        # Create test dataset and class first
        dataset_path = os.path.join(self.app.config['DATASET_PATH'], 'test_stats')
        class_path = os.path.join(dataset_path, 'test_class')
        os.makedirs(class_path, exist_ok=True)
        
        # Create a test image file
        with open(os.path.join(class_path, 'test.jpg'), 'wb') as f:
            f.write(b'test image content')
        
        # Test GET for dataset stats
        response = self.client.get('/train/datasets/test_stats/stats')
        self.assertEqual(response.status_code, 200)
        
        # Verify response is JSON with stats
        json_data = response.json
        self.assertTrue(json_data.get('success'))
        self.assertEqual(json_data.get('dataset_name'), 'test_stats')
        self.assertEqual(json_data.get('class_count'), 1)
        self.assertEqual(json_data.get('total_images'), 1)
    
    def test_get_class_images(self):
        """Test getting images for a class."""
        # Create test dataset and class first
        dataset_path = os.path.join(self.app.config['DATASET_PATH'], 'test_images')
        class_path = os.path.join(dataset_path, 'test_class')
        os.makedirs(class_path, exist_ok=True)
        
        # Create a test image file
        with open(os.path.join(class_path, 'test.jpg'), 'wb') as f:
            f.write(b'test image content')
        
        # Test GET for class images
        response = self.client.get('/train/datasets/test_images/class/test_class/images')
        self.assertEqual(response.status_code, 200)
        
        # Verify response is JSON with images
        json_data = response.json
        self.assertTrue(json_data.get('success'))
        self.assertEqual(json_data.get('class_name'), 'test_class')
        self.assertEqual(json_data.get('dataset_name'), 'test_images')
        self.assertEqual(json_data.get('image_count'), 1)
        
    def test_example_datasets_page(self):
        """Test the example datasets page route."""
        response = self.client.get('/train/datasets/example')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Example Datasets', response.data)
    
    def test_webscrape_page(self):
        """Test the webscrape page route."""
        response = self.client.get('/train/datasets/webscrape')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Web Scrape', response.data)


if __name__ == '__main__':
    unittest.main() 