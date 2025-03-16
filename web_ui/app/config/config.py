import os

class Config:
    """Base configuration for the application."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_change_in_production')
    DEBUG = False
    TESTING = False
    base_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    UPLOAD_FOLDER = os.path.join(base_path, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit to match tests
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov'}
    MODEL_PATH = os.path.join(base_path, 'models')
    DATASET_PATH = os.path.join(base_path, 'datasets')
    
    # SQLAlchemy settings
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(base_path, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    # Use SQLite for development
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(Config.base_path, 'dev.db')

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    # Use in-memory SQLite for testing
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    
    def __init__(self):
        """Initialize with environment variables."""
        # Get secret key from environment variable or use default if not set
        self.SECRET_KEY = os.environ.get('SECRET_KEY', Config.SECRET_KEY)
        # Database URI can be overridden in environment variable
        self.SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 
                                                    Config.SQLALCHEMY_DATABASE_URI)

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