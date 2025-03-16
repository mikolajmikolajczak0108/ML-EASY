"""
Unit tests for dataset utility functions.
"""
import unittest

from app import create_app
from app.modules.model_training.routes import allowed_file


class TestDatasetUtils(unittest.TestCase):
    """Test cases for dataset utility functions."""
    
    def setUp(self):
        """Set up the test environment."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def tearDown(self):
        """Tear down the test environment."""
        self.app_context.pop()
    
    def test_allowed_file(self):
        """Test the allowed_file function."""
        # Test valid extensions
        self.assertTrue(allowed_file('image.png'))
        self.assertTrue(allowed_file('image.jpg'))
        self.assertTrue(allowed_file('image.jpeg'))
        self.assertTrue(allowed_file('image.PNG'))  # Case insensitive
        
        # Test invalid extensions
        self.assertFalse(allowed_file('image.gif'))
        self.assertFalse(allowed_file('image.txt'))
        self.assertFalse(allowed_file('image'))  # No extension
        self.assertFalse(allowed_file('.png'))  # Just extension

    def test_cache_management(self):
        """Test the dataset cache management."""
        from app.modules.model_training.routes import (
            _dataset_info_cache, _class_image_cache
        )
        
        # Clear the cache for testing
        _dataset_info_cache.clear()
        _class_image_cache.clear()
        
        # Test adding to cache
        _dataset_info_cache['test_key'] = (1234, ['data'])
        self.assertIn('test_key', _dataset_info_cache)
        self.assertEqual(_dataset_info_cache['test_key'][1], ['data'])
        
        # Test cache clearing
        _dataset_info_cache.clear()
        self.assertNotIn('test_key', _dataset_info_cache)


if __name__ == '__main__':
    unittest.main() 