import os

class Config:
    """Base configuration for the application."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_change_in_production')
    DEBUG = False
    TESTING = False
    base_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    UPLOAD_FOLDER = os.path.join(base_path, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov'}
    MODEL_PATH = os.path.join(base_path, 'models')
    DATASET_PATH = os.path.join(base_path, 'datasets')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY')  # Must be set in environment for production

# Dictionary to select the appropriate configuration
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Returns the configuration based on environment variable."""
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default']) 