
#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python3; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command_exists pip3; then
    echo "pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Check if virtualenv is installed, install if not
if ! command_exists virtualenv; then
    echo "virtualenv not found. Installing virtualenv..."
    pip3 install virtualenv
fi

# Create a virtual environment
echo "Creating a virtual environment..."
virtualenv -p python3 ecotune_env

# Activate the virtual environment
echo "Activating the virtual environment..."
source ecotune_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the requirements
echo "Installing project dependencies..."
pip install -r requirements.txt

# Install the project in editable mode
echo "Installing EcoTune in editable mode..."
pip install -e .

echo "Installation complete! You can activate the virtual environment with:"
echo "source ecotune_env/bin/activate"
