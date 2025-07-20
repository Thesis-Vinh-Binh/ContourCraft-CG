conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install einops -y
conda install ffmpeg -y
conda install -c conda-forge jupyterlab -y
conda install matplotlib -y
conda install munch -y
conda install networkx -y
conda install omegaconf -y
conda install pandas pillow scikit-learn tqdm yaml -y
conda install -c iopath iopath -y

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install pyg_library torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install torch-geometric==2.4.0
pip install warp-lang

conda install cudf=24.12 cugraph=24.12 -c rapidsai
pip install smplx aitviewer chumpy scikit-image scipy trimesh loguru

git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
export CUDA_SAMPLES_INC=$(pwd)/Common
cd ..
git clone git@github.com:Dolorousrtur/CCCollisions.git
cd CCCollisions
pip install .
