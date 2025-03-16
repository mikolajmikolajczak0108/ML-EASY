"""
Routes for the home module.
"""
from flask import Blueprint, render_template

# Create blueprint
home_bp = Blueprint(
    'home', 
    __name__, 
    template_folder='../../templates/shared'
)


@home_bp.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@home_bp.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')


@home_bp.route('/contact')
def contact():
    """Render the contact page."""
    return render_template('contact.html') 