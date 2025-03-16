#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entry point for the application.
"""

import os
import sys
import subprocess
import importlib.util

# Global flag to avoid checking dependencies multiple times
dependencies_checked = False


def check_dependencies(install=True):
    """
    Check if required dependencies are installed.
    
    Args:
        install: Whether to install missing dependencies
        
    Returns:
        bool: True if all dependencies are installed
    """
    global dependencies_checked
    
    # Skip check if already done
    if dependencies_checked:
        return True
        
    try:
        # Try to find modules without actually importing them
        # to avoid potential side effects
        modules_to_check = ['flask', 'flask_sqlalchemy', 'PIL']
        missing_modules = []
        
        for module_name in modules_to_check:
            if importlib.util.find_spec(module_name) is None:
                missing_modules.append(module_name)
        
        if missing_modules:
            if not install:
                print(f"Missing modules: {', '.join(missing_modules)}")
                return False
                
            # Install missing modules
            for module_name in missing_modules:
                # Convert module name to package name if needed
                if module_name == "PIL":
                    package_name = "Pillow"
                else:
                    package_name = module_name
                    
                print(f"Installing {package_name}...")
                try:
                    # Use --quiet for less verbose output
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        package_name, "--quiet"
                    ])
                except subprocess.CalledProcessError:
                    print(f"Failed to install {package_name}")
                    return False
            
            # Verify all modules are now installed
            return check_dependencies(install=False)
        
        # All modules are available
        dependencies_checked = True
        return True
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        return False


def run_app():
    """Run the Flask application."""
    from app import create_app
    
    # Create application instance
    app = create_app()
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True  # Enable threading for better performance
    )


if __name__ == '__main__':
    print("=== ML-EASY - Machine Learning Made Easy ===")
    
    # Check for old app.py file and rename it if found
    if os.path.exists('app.py') and not os.path.exists('app.py.bak'):
        os.rename('app.py', 'app.py.bak')
        print("Renamed old app.py to app.py.bak")
    
    # Only check dependencies on first run
    if not os.path.exists('.dependencies_installed'):
        # Check dependencies
        if check_dependencies():
            # Create a marker file to skip dependency check next time
            with open('.dependencies_installed', 'w') as f:
                f.write('1')
            print("All dependencies installed successfully.")
        else:
            print("Please install the required dependencies manually.")
            sys.exit(1)
    
    # Run the application
    run_app()