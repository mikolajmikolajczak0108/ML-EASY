"""
Integration tests for the application routes.
"""
import os
import unittest
from app import create_app


class TestRoutes(unittest.TestCase):
    """Test cases for the application routes."""
    
    def setUp(self):
        """Set up the test environment."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
    def test_home_page(self):
        """Test the home page route."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'ML-EASY', response.data)
        self.assertIn(b'Machine Learning Made Easy', response.data)
        
    def test_test_models_page(self):
        """Test the test models page route."""
        response = self.client.get('/test/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Test Models', response.data)
        
    def test_train_models_page(self):
        """Test the train models page route."""
        response = self.client.get('/train/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Train Models', response.data)
        
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
        
    def test_example_datasets_page(self):
        """Test the example datasets page route."""
        response = self.client.get('/train/datasets/example')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Example Datasets', response.data)


if __name__ == '__main__':
    unittest.main() 