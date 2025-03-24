#!/bin/bash

# One-stop script to:
#   1. Install Miniconda.
#   2. Install packages from environment.yml into the base environment.
#   3. Append a project directory to PYTHONPATH.

# Usage in Colab:
#   !chmod +x setup_colab.sh
#   !./setup_colab.sh

# ============================================

# 1) Download & Install Miniconda (x86_64 build for Colab)
MINICONDA_INSTALLER_SCRIPT=Miniconda3-latest-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local/miniconda

# Download Miniconda
wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT

# Install Miniconda
bash ./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX

# Initialize conda in this shell
eval "$($MINICONDA_PREFIX/bin/conda shell.bash hook)"
conda config --set auto_activate_base false

# ============================================

# 2) Install packages from environment.yml into the base environment
if [ -f "environment.yml" ]; then
    echo "Found environment.yml. Installing packages into the base environment..."
    conda env update -n base -f environment.yml
else
    echo "ERROR: environment.yml not found!"
    exit 1
fi

# ============================================

# 3) Append a Project Directory to PYTHONPATH
PROJECT_PATH="/content/drive/Othercomputers/kyei_mac_m3_pro/MediMaven"

# Append to PYTHONPATH in the current shell
if [ -d "$PROJECT_PATH" ]; then
    echo "Appending $PROJECT_PATH to PYTHONPATH..."
    export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"
    
    # Append to conda's environment activation script for persistence
    echo "export PYTHONPATH=\"$PROJECT_PATH:\$PYTHONPATH\"" >> $MINICONDA_PREFIX/etc/profile.d/conda.sh
else
    echo "ERROR: Project directory $PROJECT_PATH not found!"
    exit 1
fi

echo "Setup complete. You can now import modules from your project directory."