#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Initialize Conda in the current shell if not already initialized
if ! command -v conda &> /dev/null; then
    echo "Conda not initialized. Run 'conda init' for your shell and restart."
    exit 1
fi

# Create a conda environment
ENV_NAME="vlfm_orig"
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.9 -y

# Ensure conda is properly activated
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch with CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install numpy
pip install numpy==1.26.4

# Install Habitat-Sim
echo "Installing Habitat-Sim v0.2.4"
conda install habitat-sim=0.2.4 -c conda-forge -c aihabitat -y

# Clone and install Habitat-Lab and Habitat-Baselines
echo "Cloning and installing Habitat-Lab and Habitat-Baselines v0.2.4"
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..



# Install Salesforce LAVIS
echo "Installing Salesforce LAVIS"
pip install salesforce-lavis==1.0.2

# Install VLFM
echo "Cloning and installing VLFM"
git clone git@github.com:bdaiinstitute/vlfm.git  # Replace with the actual VLFM repository URL
cd vlfm
git clone git@github.com:WongKinYiu/yolov7.git

# Modify pyproject.toml to remove the torch-related dependencies
sed -i "/torch ==/d" pyproject.toml
sed -i "/torchvision ==/d" pyproject.toml

# Install VLFM
pip install -e .


# Install GroundingDINO from source
echo "Cloning and installing GroundingDINO"
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
cd ..
cd ..

echo "Downloading additional models and weights into the 'data' folder"
# mkdir -p VLFM/data
wget -q https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt -P vlfm/data
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P vlfm/data
wget -q https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt -P vlfm/data

# # Reinstall PyTorch and numpy to ensure compatibility
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4

# Completion message
echo "Environment setup complete. Activate it using 'conda activate $ENV_NAME'"
