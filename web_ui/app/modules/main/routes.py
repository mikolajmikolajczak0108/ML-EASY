"""
Routes for the main module.
"""
from flask import Blueprint, render_template, redirect, url_for

# Create blueprint
main_bp = Blueprint(
    'main', 
    __name__, 
    template_folder='../../templates/shared'
)


@main_bp.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@main_bp.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')


@main_bp.route('/contact')
def contact():
    """Render the contact page."""
    return render_template('contact.html') 