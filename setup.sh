#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Create config.toml with proper settings
cat > .streamlit/config.toml <<EOL
[server]
headless = true
port = 8501
enableCORS = true
enableXsrfProtection = false

[browser]
serverAddress = "0.0.0.0"

[runner]
fastReruns = true

[logger]
level = "info"

[client]
showErrorDetails = true
EOL

# Install Python packages with specific versions
pip3 install --no-cache-dir -r requirements.txt

# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging
export PYTHONPATH="${PYTHONPATH}:/app"

echo "Setup completed successfully"
