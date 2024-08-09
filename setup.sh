#!/bin/bash

setup_venv() {
    echo "Setting up venv..."

    python -m venv venv
    source venv/bin/activate

    echo "Done setting up venv!"
}

install_requirements() {
    echo "Installing requirements..."

    yes | pip install -r requirements.txt --upgrade

    echo "Done installing requirements!"
}

echo "Running set up..."

setup_venv
install_requirements

echo "All set up!"