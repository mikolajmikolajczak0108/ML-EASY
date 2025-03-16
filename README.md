# ML-EASY: Machine Learning Made Easy

A user-friendly web application for creating, training, and using machine learning models without coding expertise.

![ML-EASY Screenshot](https://github.com/mikolajmikolajczak0108/ML-EASY/raw/master/web_ui/app/static/images/screenshot.png)

## Overview

ML-EASY simplifies the machine learning workflow by providing an intuitive web interface for:
- Creating and managing datasets
- Training machine learning models
- Testing and evaluating models
- Applying models to new data

## Features

- **Dataset Management**
  - Create custom datasets with multiple classes
  - Upload and organize images for each class
  - Edit and maintain datasets through a visual interface
  
- **Model Training**
  - Train image classification models with a few clicks
  - Choose from various pre-configured architectures
  - Fine-tune existing models
  
- **Testing & Evaluation**
  - Test models on new images
  - Batch classify multiple files
  - View performance metrics and evaluation results

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/mikolajmikolajczak0108/ML-EASY.git
   cd ML-EASY/ML-EASY/web_ui
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python run.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

### Creating a Dataset

1. Navigate to "Model Training" > "Datasets"
2. Click "Create New Dataset"
3. Enter dataset name and number of classes
4. Upload images to each class

### Training a Model

1. Go to "Model Training" > "Train New Model"
2. Select a dataset
3. Choose model architecture and parameters
4. Start training

### Testing Your Model

1. Navigate to "Model Testing"
2. Select your trained model
3. Upload images for classification
4. View results

## Recent Improvements

- Enhanced dataset management with improved error handling
- Fixed UI issues with upload modals
- Added robust debugging tools
- Improved server-side file processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Thanks to all contributors who have helped improve this application
- Special thanks to the open-source machine learning community 