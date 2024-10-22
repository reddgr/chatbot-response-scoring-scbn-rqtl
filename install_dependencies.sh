#!/bin/bash

# Update package list and install pip if not already installed
if ! command -v pip3 &> /dev/null
then
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Install dependencies from requirements.txt
pip install -r requirements.txt