"""
Initialize the Flask application with a factory pattern.
"""
import os
from flask import Flask

from app.config.config import get_config


def create_app(config_name=None):
    """
    Application factory function that creates and configures the Flask app.
    
    Args:
        config_name: Configuration to use (development, testing, production)
        
    Returns:
        Configured Flask application
    """
    # Create and configure the app
    app = Flask(__name__)
    
    # Load the configuration
    config_obj = get_config()
    app.config.from_object(config_obj)
    
    # Ensure the required directories exist
    create_required_directories(app)
    
    # Register blueprints
    register_blueprints(app)
    
    return app


def create_required_directories(app):
    """
    Create the required directories for the application.
    
    Args:
        app: Flask application instance
    """
    # Create required directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_PATH'], exist_ok=True)
    os.makedirs(app.config['DATASET_PATH'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'processed'), 
                exist_ok=True)


def register_blueprints(app):
    """
    Register blueprints with the application.
    
    Args:
        app: Flask application instance
    """
    # Import modules/blueprints
    from app.modules.main.routes import main_bp
    from app.modules.model_testing.routes import model_testing_bp
    from app.modules.model_training.routes import model_training_bp
    
    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(model_testing_bp)
    app.register_blueprint(model_training_bp) 