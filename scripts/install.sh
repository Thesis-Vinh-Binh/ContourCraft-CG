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


conda create -n ccraft python=3.10
conda activate ccraft

export REPO_DIR=$(pwd)
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.6"

pip install torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install torchvision==0.19.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
conda install einops ffmpeg -y

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" # Takes a while to build wheel


# conda install -c conda-forge jupyterlab
conda install matplotlib munch omegaconf pillow scikit-learn yaml -y

pip install cudf-cu12 torch-geometric==2.4.0 smplx trimesh easydict

pip install pyg-library torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
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
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
export CUDA_SAMPLES_INC=$(pwd)/Common
cd ..
git clone git@github.com:Dolorousrtur/CCCollisions.git
cd CCCollisions
pip install .

pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/
apt update && apt install -y libsm6 libxext6 libegl1 libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 libgl1-mesa-glx libopengl0

cd $REPO_DIR
export PYTHONPATH=$(pwd)
python simulation_example.py 

wget https://github.com/Meshcapade/SMPL_blender_addon/archive/refs/heads/main.zip -O smplx_addon.zip
unzip smplx_addon.zip
mv SMPL_blender_addon-main/ /workspace/blender-3.6.14-linux-x64/3.6/scripts/addons/smplx_blender_addon
cp -r /workspace/blender-3.6.14-linux-x64/3.6/scripts/addons/smplx_blender_addon /venv/ccraft/lib/python3.10/site-packages/bpy/3.6/scripts/addons

/workspace/blender-3.6.14-linux-x64/3.6/python/bin/python3.10 -m ensurepip
/workspace/blender-3.6.14-linux-x64/3.6/python/bin/python3.10 -m pip install pyyaml tqdm