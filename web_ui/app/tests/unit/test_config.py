"""
Unit tests for the configuration module.
"""
import os
import unittest
from app.config.config import Config, DevelopmentConfig, TestingConfig, ProductionConfig


class TestConfig(unittest.TestCase):
    """Test cases for the configuration module."""
    
    def test_base_config(self):
        """Test the base configuration."""
        self.assertEqual(Config.DEBUG, False)
        self.assertEqual(Config.TESTING, False)
        self.assertIn('uploads', Config.UPLOAD_FOLDER)
        self.assertIn('models', Config.MODEL_PATH)
        self.assertIn('datasets', Config.DATASET_PATH)
        self.assertEqual(Config.MAX_CONTENT_LENGTH, 16 * 1024 * 1024)
        self.assertIn('png', Config.ALLOWED_EXTENSIONS)
        self.assertIn('jpg', Config.ALLOWED_EXTENSIONS)
        self.assertIn('jpeg', Config.ALLOWED_EXTENSIONS)
        self.assertIn('gif', Config.ALLOWED_EXTENSIONS)
        self.assertIn('mp4', Config.ALLOWED_EXTENSIONS)
        self.assertIn('mov', Config.ALLOWED_EXTENSIONS)
    
    def test_development_config(self):
        """Test the development configuration."""
        self.assertEqual(DevelopmentConfig.DEBUG, True)
        self.assertEqual(DevelopmentConfig.TESTING, False)
    
    def test_testing_config(self):
        """Test the testing configuration."""
        self.assertEqual(TestingConfig.DEBUG, True)
        self.assertEqual(TestingConfig.TESTING, True)
    
    def test_production_config(self):
        """Test the production configuration."""
        self.assertEqual(ProductionConfig.DEBUG, False)
        self.assertEqual(ProductionConfig.TESTING, False)
        
        # Test that SECRET_KEY is set from environment variable
        os.environ['SECRET_KEY'] = 'test_secret_key'
        self.assertEqual(ProductionConfig.SECRET_KEY, 'test_secret_key')
        
        # Clean up
        del os.environ['SECRET_KEY']


if __name__ == '__main__':
    unittest.main() 