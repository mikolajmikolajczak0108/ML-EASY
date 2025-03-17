"""
Flask application package.
"""
from flask import Flask, send_from_directory, request, render_template
from app.config.config import Config
import os
import sys
import traceback
import logging
from flask_sqlalchemy import SQLAlchemy
import time

# Konfiguracja logowania
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize database
db = SQLAlchemy()

# Cache for directories to avoid checking each time
_directories_created = False

# Cache for database initialization
_db_initialized = False


def create_app(config_class=Config):
    """
    Create and configure the Flask application.
    
    Args:
        config_class: Configuration class to use
        
    Returns:
        app: Configured Flask application
    """
    global _directories_created, _db_initialized
    
    start_time = time.time()
    
    app = Flask(
        __name__, 
        template_folder='templates',  # Jawnie wskazujemy folder templates
        static_folder='static'        # Jawnie wskazujemy folder static
    )
    app.config.from_object(config_class)
    
    # Ensure directories exist (only once)
    if not _directories_created:
        _ensure_directories(app)
        _directories_created = True
    
    # Initialize extensions
    db.init_app(app)
    
    # Register commands
    from app.commands import register_commands
    register_commands(app)
    
    # Create database tables if they don't exist (only once)
    if not _db_initialized:
        with app.app_context():
            db_path = app.config['SQLALCHEMY_DATABASE_URI'].split('/')[-1]
            if db_path == ':memory:' or not os.path.exists(db_path):
                try:
                    db.create_all()
                    _db_initialized = True
                    logger.info("Database tables created.")
                except Exception as e:
                    logger.error(f"Warning: Could not create database tables: {e}")
    
    # Register blueprints
    _register_blueprints(app)
    
    # Set up static routes for datasets and uploads
    _setup_static_routes(app)
    
    # Register error handlers
    _register_error_handlers(app)
    
    # Register request/response loggers
    _register_request_handlers(app)
    
    # Preload ML libraries for faster training
    @app.before_first_request
    def preload_ml_libraries():
        """Preload ML libraries to avoid importing them during training."""
        app.logger.info("Pre-loading ML libraries for model training...")
        try:
            # Import libraries in a separate thread to avoid blocking app startup
            import threading
            
            def import_libraries():
                try:
                    import torch
                    app.logger.info(f"PyTorch loaded: {torch.__version__}")
                    
                    # Import fastai vision library
                    from fastai.vision.all import ImageDataLoaders, vision_learner
                    app.logger.info("fastai.vision.all loaded")
                    
                    # Import fastai metrics
                    from fastai.metrics import Precision, Recall, F1Score
                    app.logger.info("fastai.metrics loaded")
                    
                    # Import torchvision models
                    import torchvision.models as models
                    app.logger.info("torchvision.models loaded")
                    
                    app.logger.info("All ML libraries pre-loaded successfully")
                except Exception as e:
                    app.logger.error(f"Error pre-loading ML libraries: {e}")
            
            # Start the thread for library import
            import_thread = threading.Thread(target=import_libraries)
            import_thread.daemon = True
            import_thread.start()
            app.logger.info("ML library preloading started in background")
            
        except Exception as e:
            app.logger.error(f"Failed to start ML library preloading: {e}")
    
    # Print startup time for optimization tracking
    logger.info(f"Application initialized in {time.time() - start_time:.2f} seconds")
    
    return app


def _ensure_directories(app):
    """Create all required directories and symlinks."""
    # Create required directories
    required_dirs = [
        app.config['UPLOAD_FOLDER'],
        app.config['MODEL_PATH'],
        app.config['DATASET_PATH'],
        os.path.join(app.config['UPLOAD_FOLDER'], 'processed')
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Create dataset static access point (once)
    dataset_static_path = os.path.join(app.static_folder, 'datasets')
    if not os.path.exists(dataset_static_path):
        try:
            # On Windows, might need special permissions, so handle errors
            if os.name == 'nt':  # Windows
                # On Windows, try direct copy instead
                if not os.path.exists(dataset_static_path):
                    os.makedirs(dataset_static_path, exist_ok=True)
            else:  # Linux/Mac
                # Create a symlink on Unix systems
                target_path = app.config['DATASET_PATH']
                # Use relative path if possible
                if os.path.exists(target_path):
                    os.symlink(target_path, dataset_static_path, 
                              target_is_directory=True)
                    
            logger.info(f"Created dataset access path at {dataset_static_path}")
        except Exception as e:
            logger.error(f"Warning: Could not create dataset access path: {e}")


def _register_blueprints(app):
    """Register all application blueprints efficiently."""
    from app.modules.home.routes import home_bp
    from app.modules.model_training.routes import model_training_bp
    from app.modules.model_testing.routes import model_testing_bp
    
    app.register_blueprint(home_bp)
    app.register_blueprint(model_training_bp, url_prefix='/train')
    app.register_blueprint(model_testing_bp, url_prefix='/test')
    
    logger.info("Blueprints registered successfully")


def _setup_static_routes(app):
    """Set up routes for serving static files."""
    @app.route('/static/datasets/<path:filename>')
    def serve_dataset(filename):
        return send_from_directory(app.config['DATASET_PATH'], filename)
    
    @app.route('/static/uploads/<path:filename>')
    def serve_upload(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def _register_error_handlers(app):
    """Register error handlers for the application."""
    
    @app.errorhandler(404)
    def page_not_found(e):
        logger.warning(f"404 Error: {request.path} - {e}")
        return render_template('error.html', error=f"Page not found: {request.path}", code=404), 404
    
    @app.errorhandler(500)
    def internal_server_error(e):
        # Log the exception details
        logger.error(f"500 Error: {request.path}")
        logger.error(f"Exception: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Always return a proper error page
        error_message = str(e) if app.debug else "Internal Server Error"
        return render_template('error.html', error=error_message, code=500), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        # Log the exception details
        logger.error(f"Unhandled Exception: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Always return a proper error page
        error_message = str(e) if app.debug else "An unexpected error occurred"
        return render_template('error.html', error=error_message, code=500), 500


def _register_request_handlers(app):
    """Register request and response handlers for logging."""
    
    @app.before_request
    def log_request():
        logger.debug(f"Request: {request.method} {request.path}")
        if request.args:
            logger.debug(f"Request args: {dict(request.args)}")
        if request.form:
            logger.debug(f"Request form: {dict(request.form)}")
        if request.is_json:
            logger.debug(f"Request JSON: {request.json}")
    
    @app.after_request
    def log_response(response):
        logger.debug(f"Response: {response.status}")
        return response 