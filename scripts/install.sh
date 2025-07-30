#!/bin/bash
#SBATCH --job-name=setup_CG
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # more threads may assist loading
#SBATCH --mem=32G                       # host RAM for data processing
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/setup_%j.out  # %j is the job ID
#SBATCH --error=logs/setup_%j.out   # (optional) separate stderr

# module purge
# module load anaconda3/2023.9
conda init
source ~/.bashrc

export REPO_DIR=$(pwd)

conda create -n ccraft python=3.10
conda activate ccraft

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.6"

pip install torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install torchvision==0.19.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
conda install einops
conda install ffmpeg

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" # Takes a while to build wheel


conda install -c conda-forge jupyterlab
conda install matplotlib
conda install munch
# conda install networkx
conda install omegaconf
conda install pandas pillow scikit-learn tqdm yaml
conda install -c iopath iopath

pip install cudf-cu12
pip install torch-geometric==2.4.0
pip install smplx
pip install trimesh

pip install pyg-library torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
# pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" # Takes a while to build wheel

pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com

pip install gdown 
gdown 1QXezA3J6uXqWHGATmcw3jaYxRXY2Ctte 
unzip assets.zip
gdown 1NfxAeaC2va8TWMjiO_gbAcVPnZ8BYFPD
unzip ccraft_data.zip
cd ccraft_data/aux_data/body_models
gdown 1Ooo9IWcHdTKzlDSk-CiRz5oSTrlCeutV
unzip models_smplx_v1_1.zip

cd $REPO_DIR

pip install warp-lang
cd cuda-samples
export CUDA_SAMPLES_INC=$(pwd)/Common
cd ..
cd CCCollisions
pip install .
cd ..
pip install bpy
pip install easydict
