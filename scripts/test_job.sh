#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # more threads may assist loading
#SBATCH --mem=32G                       # host RAM for data processing
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out  # %j is the job ID
#SBATCH --error=logs/slurm_%j.out   # (optional) separate stderr

# module purge
# module load anaconda3/2023.9
conda init
source ~/.bashrc
conda activate ccraft

export PYTHONPATH=$(pwd)

# python -c "import torch; print(torch.cuda.device_count());"

python simulation_example.py 

# python get_smplx_mesh.py # get smplx mesh
# python get_mesh_from_simulation.py # get garment mesh