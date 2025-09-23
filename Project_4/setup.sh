#!/bin/bash

# Setup script for the Autoencoder-Stacked-Ensemble Pipeline

echo "Setting up Autoencoder-Stacked-Ensemble Pipeline..."
echo "=================================================="

# Check Python version
python_version=$(python --version 2>&1)
echo "Python version: $python_version"

# Check if Python 3.7+ is available
if ! python -c "import sys; assert sys.version_info >= (3, 7)" 2>/dev/null; then
    echo "Error: Python 3.7 or higher is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p results
mkdir -p data

echo "Setup completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To run the demo:"
echo "python demo.py"
echo ""
echo "To run the full pipeline:"
echo "python main.py --config config/config.yaml"
