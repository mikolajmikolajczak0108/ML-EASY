"""
Flask CLI commands for the application.
"""
import click
from flask.cli import with_appcontext
from app import db


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Initialize the database."""
    # Create all tables
    db.create_all()
    click.echo('Database initialized.')


def register_commands(app):
    """Register custom Flask CLI commands.
    
    Args:
        app: Flask application instance
    """
    app.cli.add_command(init_db_command) 