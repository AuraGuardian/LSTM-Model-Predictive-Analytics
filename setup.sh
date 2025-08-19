#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages with specific versions
pip3 install --no-cache-dir -r requirements.txt

# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging
export PYTHONPATH="${PYTHONPATH}:/app"
