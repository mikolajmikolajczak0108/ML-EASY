"""
Unit tests for dataset routes.
"""
import unittest
from unittest.mock import patch, MagicMock

from app import create_app


class TestDatasetRouteUnit(unittest.TestCase):
    """Test cases for dataset routes using mocks."""
    
    def setUp(self):
        """Set up the test environment."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
        self.app.config['DATASET_PATH'] = '/fake/path'
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        
        # Dodajemy szablon error.html, aby uniknąć błędów
        with open('app/templates/error.html', 'w') as f:
            f.write("""
            {% extends "base.html" %}
            {% block title %}Error{% endblock %}
            {% block content %}
            <div class="container mt-5">
                <div class="alert alert-danger">
                    <h4 class="alert-heading">Error {{ code }}</h4>
                    <p>{{ error }}</p>
                </div>
            </div>
            {% endblock %}
            """)
    
    def tearDown(self):
        """Tear down the test environment."""
        self.app_context.pop()
        
        # Usuwamy tymczasowy plik szablonu
        import os
        if os.path.exists('app/templates/error.html'):
            os.remove('app/templates/error.html')
    
    @patch('app.modules.model_training.routes.render_template')
    @patch('app.modules.model_training.routes.logger')
    def test_index(self, mock_logger, mock_render):
        """Test the index function."""
        mock_render.return_value = 'rendered template'
        
        response = self.client.get('/train/')
        
        mock_logger.info.assert_called_once()
        mock_render.assert_called_once_with('model_training/train_index.html')
        
    @patch('app.modules.model_training.routes.render_template')
    @patch('app.modules.model_training.routes.logger')
    def test_new_model(self, mock_logger, mock_render):
        """Test the new_model function."""
        mock_render.return_value = 'rendered template'
        
        response = self.client.get('/train/new-model')
        
        mock_logger.info.assert_called_once()
        mock_render.assert_called_once_with('model_training/new_model.html')
    
    @patch('app.modules.model_training.routes.render_template')
    @patch('app.modules.model_training.routes.logger')
    def test_finetune(self, mock_logger, mock_render):
        """Test the finetune function."""
        mock_render.return_value = 'rendered template'
        
        response = self.client.get('/train/finetune')
        
        mock_logger.info.assert_called_once()
        mock_render.assert_called_once_with('model_training/finetune.html')
    
    @patch('app.modules.model_training.routes.os.path.exists')
    @patch('app.modules.model_training.routes.os.listdir')
    @patch('app.modules.model_training.routes.render_template')
    @patch('app.modules.model_training.routes.logger')
    @patch('app.modules.model_training.routes._dataset_info_cache', {})
    def test_datasets_fresh_data(self, mock_logger, mock_render, 
                                mock_listdir, mock_exists):
        """Test the datasets function with fresh data."""
        mock_exists.return_value = True
        mock_listdir.return_value = ['dataset1', 'dataset2']
        
        # Mock nested listdir for classes
        def side_effect(path):
            if 'dataset1' in path or 'dataset2' in path:
                return ['class1', 'class2']
            return ['dataset1', 'dataset2']
        
        mock_listdir.side_effect = side_effect
        
        # Mock os.path.join
        with patch('app.modules.model_training.routes.os.path.join') as mock_join:
            mock_join.side_effect = lambda *args: '/'.join(args)
            
            # Mock os.path.isdir
            with patch('app.modules.model_training.routes.os.path.isdir') as mock_isdir:
                mock_isdir.return_value = True
                
                mock_render.return_value = 'rendered template'
                
                response = self.client.get('/train/datasets')
                
                mock_logger.info.assert_called_once()
                mock_render.assert_called_once()
    
    @patch('app.modules.model_training.routes.os.path.exists')
    @patch('app.modules.model_training.routes.os.makedirs')
    @patch('app.modules.model_training.routes.logger')
    def test_create_dataset_post(self, mock_logger, mock_makedirs, mock_exists):
        """Test the create_dataset function with POST."""
        # Setup mocks
        mock_exists.return_value = False
        
        data = {
            'dataset_name': 'test_dataset',
            'num_classes': '2',
            'class_names': ['class1', 'class2']
        }
        
        response = self.client.post('/train/datasets/new', data=data)
        
        mock_logger.info.assert_called()
        mock_makedirs.assert_called()
        self.assertEqual(response.status_code, 200)
    
    @patch('app.modules.model_training.routes.render_template')
    @patch('app.modules.model_training.routes.logger')
    def test_create_dataset_get(self, mock_logger, mock_render):
        """Test the create_dataset function with GET."""
        mock_render.return_value = 'rendered template'
        
        response = self.client.get('/train/datasets/new')
        
        mock_logger.info.assert_called_once()
        mock_render.assert_called_once_with('model_training/new_dataset.html')
    
    @patch('app.modules.model_training.routes.render_template')
    @patch('app.modules.model_training.routes.logger')
    def test_edit_dataset(self, mock_logger, mock_render):
        """Test the edit_dataset function."""
        from app.modules.model_training.routes import model_training_bp
        
        # Sprawdzamy, czy funkcja edit_dataset jest zarejestrowana w blueprincie
        routes = [rule for rule in self.app.url_map.iter_rules() 
                 if rule.endpoint.startswith('model_training.')]
        
        # Sprawdzamy, czy istnieje ścieżka dla edit_dataset
        edit_dataset_route = [r for r in routes 
                             if r.endpoint == 'model_training.edit_dataset']
        
        self.assertTrue(len(edit_dataset_route) > 0, 
                       "Nie znaleziono ścieżki dla edit_dataset")
        
        # Sprawdzamy, czy ścieżka ma prawidłowy format
        route = edit_dataset_route[0]
        self.assertIn('/datasets/<dataset_name>/edit', route.rule)
    
    @patch('app.modules.model_training.routes.jsonify')
    @patch('app.modules.model_training.routes.os.path.exists')
    @patch('app.modules.model_training.routes.os.makedirs')
    @patch('app.modules.model_training.routes.logger')
    @patch('app.modules.model_training.routes._dataset_info_cache', {'dataset_test': 'value'})
    def test_new_class(self, mock_logger, mock_makedirs, mock_exists, mock_jsonify):
        """Test the new_class function."""
        from app.modules.model_training.routes import new_class
        
        # Setup mocks
        mock_exists.return_value = False
        mock_jsonify.return_value = {'success': True}
        
        # Mock os.path.join
        with patch('app.modules.model_training.routes.os.path.join') as mock_join:
            mock_join.side_effect = lambda *args: '/'.join(args)
            
            # Używamy kontekstu żądania z danymi formularza
            with self.app.test_request_context(
                '/train/datasets/test_dataset/classes/new',
                method='POST',
                data={'class_name': 'test_class'}
            ):
                result = new_class('test_dataset')
                
                mock_logger.info.assert_called()
                mock_makedirs.assert_called_once()
                self.assertEqual(result, {'success': True})
    
    @patch('app.modules.model_training.routes.render_template')
    @patch('app.modules.model_training.routes.logger')
    def test_example_datasets(self, mock_logger, mock_render):
        """Test the example_datasets function."""
        mock_render.return_value = 'rendered template'
        
        response = self.client.get('/train/datasets/example')
        
        mock_logger.info.assert_called_once()
        mock_render.assert_called_once()
    
    @patch('app.modules.model_training.routes.render_template')
    @patch('app.modules.model_training.routes.logger')
    def test_webscrape(self, mock_logger, mock_render):
        """Test the webscrape function."""
        mock_render.return_value = 'rendered template'
        
        response = self.client.get('/train/datasets/webscrape')
        
        mock_logger.info.assert_called_once()
        mock_render.assert_called_once_with('model_training/webscrape.html')


if __name__ == '__main__':
    unittest.main() 