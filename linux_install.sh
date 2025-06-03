#!/bin/bash

# Conda setup
source ~/anaconda3/etc/profile.d/conda.sh

# clone the repo
git clone https://github.com/florataly/Chimpanzee-Pose-Estimation-Comparison.git
cd Chimpanzee-Pose-Estimation-Comparison

# Create missing directories
mkdir -p predictions data/ground_truth data/TEST evaluation/results temp

### OPENAPEPOSE
cd OpenApePose

# Set up MMPose environment
conda activate
conda create -n MMPose026 python=3.8 -y
conda activate MMPose026
conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3 -c pytorch

# Install mmcv-full (wheel for torch 1.10.2 + cu113)
pip install openmim==0.3.3
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

git clone --branch v0.26.0 https://github.com/open-mmlab/mmpose.git
cd mmpose
mkdir -p checkpoints mmpose/datasets/datasets/oap
pip install -e .
cd ..

# Install MMDetection from source
git clone --branch v2.25.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
cd ../..

# Install required Python packages
pip install tomli==2.0.1 platformdirs==3.5.1

# Downloading OpenApePose
cd temp

# downloading google drive folders
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1KG-1eaxqK0WGjRTCYDpOxNQa5XH5bqJr
gdown --folder https://drive.google.com/drive/folders/1KO9D056XhdorgHr8UIuTS-yRz2euUrll

cd code
mv hrnet_w48_oap_256x192_full.py ../../OpenApePose/mmpose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/macaque
mv TopDownOAPDataset.py ../../OpenApePose/mmpose/configs/_base_/datasets
mv animal_oap_dataset.py ../../OpenApePose/mmpose/mmpose/datasets/datasets/oap
sed -i "/^from \.omc import (TopDownOMCDataset)/d" __init__.py # removing line that would error otherwise
mv __init__.py ../../OpenApePose/mmpose/mmpose/datasets/datasets

# Write __init__.py script for animal_oap_dataset.py
cat <<EOF > __init__.py
#!/usr/bin/env python3
from .animal_oap_dataset import TopDownOAPDataset
__all__ = [ 'TopDownOAPDataset']
EOF
mv __init__.py ../../OpenApePose/mmpose/mmpose/datasets/datasets/oap

cd ../models
mv hrnet_w48_oap_256x192_full.pth ../../OpenApePose/mmpose/checkpoints

cd ../..
rmdir temp/code temp/models temp

cd ..
conda deactivate

### DEEPWILD
conda create -n DEEPLABCUT "python=3.10" -y
conda activate DEEPLABCUT

# Install the desired TensorFlow version, built for CUDA 11.8 and cuDNN 8
pip install "tensorflow==2.12" "tensorpack>=0.11" "tf_slim>=1.1.0" tensorrt-cu11

# Install PyTorch with a version using CUDA 11.8 and cuDNN 8
pip install "torch==2.3.1" torchvision --index-url https://download.pytorch.org/whl/cu118

# Create symbolic links to NVIDIA shared libraries for TensorFlow
#   -> as described in their installation docs:
#      https://www.tensorflow.org/install/pip#step-by-step_instructions

pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
ln -svf ../nvidia/*/lib/*.so* .
popd

pip install  --pre deeplabcut[gui]
pip install statsmodels matplotlib seaborn

conda install -c conda-forge ipywidgets

conda deactivate