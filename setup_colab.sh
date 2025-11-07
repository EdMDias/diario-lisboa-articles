#!/bin/bash
# Setup colab-cli with local config for this project only

set -e

echo "=========================================="
echo "Setting up colab-cli for this project"
echo "=========================================="

# Set local config directory
export COLAB_CLI_CONFIG_DIR="$(pwd)/.colab-config"
mkdir -p "$COLAB_CLI_CONFIG_DIR"

echo ""
echo "Local config directory: $COLAB_CLI_CONFIG_DIR"

# Check if credentials exist
if [ ! -f ".credentials/client_secrets.json" ]; then
    echo ""
    echo "❌ Error: client_secrets.json not found in .credentials/"
    echo ""
    echo "Please:"
    echo "1. Go to: https://console.developers.google.com/"
    echo "2. Create/select project"
    echo "3. Enable Google Drive API"
    echo "4. Create OAuth 2.0 credentials (Desktop app)"
    echo "5. Download as .credentials/client_secrets.json"
    exit 1
fi

echo ""
echo "✅ Found client_secrets.json"

# Set config
echo ""
echo "Configuring colab-cli..."
COLAB_CLI_CONFIG_DIR="$COLAB_CLI_CONFIG_DIR" colab-cli set-config .credentials/client_secrets.json

# Set auth user
echo ""
echo "Setting up authentication..."
echo "A browser will open - please log in with eduardo.m.dias@gmail.com"
echo ""
read -p "Press Enter to continue..."

COLAB_CLI_CONFIG_DIR="$COLAB_CLI_CONFIG_DIR" colab-cli set-auth-user 0

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To use colab-cli in this project, run commands with:"
echo "  ./colab open-nb colab_ocr_notebook.ipynb"
echo ""
echo "Or manually set the config directory:"
echo "  export COLAB_CLI_CONFIG_DIR=$(pwd)/.colab-config"
echo "  colab-cli open-nb colab_ocr_notebook.ipynb"
