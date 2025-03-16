"""
Database models for the application.
"""
from datetime import datetime
from app import db


class Dataset(db.Model):
    """Model representing a dataset for machine learning."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Define the relationship with classes
    classes = db.relationship('DatasetClass', backref='dataset', lazy=True)
    
    def __repr__(self):
        return f'<Dataset {self.name}>'


class DatasetClass(db.Model):
    """Model representing a class within a dataset."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    image_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<DatasetClass {self.name}>'


class TrainedModel(db.Model):
    """Model representing a trained machine learning model."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    architecture = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    file_path = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Define the relationship with datasets
    dataset = db.relationship('Dataset', backref='models')
    
    def __repr__(self):
        return f'<TrainedModel {self.name}>' 