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

conda env create -f  ccraft.yml
conda activate contourcraft

export FORCE_CUDA=1

pip install pyg-library torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" # Takes a while to build wheel

cd cuda-samples
export CUDA_SAMPLES_INC=$(pwd)/Common
cd ..
cd CCCollisions
pip install .
cd ..
pip install bpy
pip install easydict
