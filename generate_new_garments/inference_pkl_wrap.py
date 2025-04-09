import argparse
import time
import subprocess
import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, required=True, help="path to the save resules shapenet dataset")
parser.add_argument("--index2", type=int, default=0, help="path to the save resules shapenet dataset")
args = parser.parse_args()



def run_blender_render(index):
    process = subprocess.Popen(
        ["/is/cluster/fast/sbian/anaconda3/envs/hood0/bin/python", 
        "generate_new_garments/inference_pkl.py", "--index", str(index),], stdout=subprocess.PIPE
    )

    process.wait()
    print('finished', index, process.returncode)
    return

def main():
    args.index = args.index * 4 + args.index2
    print('args.index', args.index)

    # sampled_clothes_list_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/sampled_clothes_v3_combine_all.pkl'
    sampled_clothes_list_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/sampled_clothes_v4_combine_all.pkl'
    with open(sampled_clothes_list_path, 'rb') as f:
        sampled_clothes_list = pkl.load(f)

    # sampled_clothes_list = random.shuffle(sampled_clothes_list)
    sampled_clothes_list_len = len(sampled_clothes_list)
    sample_per_process = sampled_clothes_list_len // 800 + 1
    all_indices = np.arange(sampled_clothes_list_len)

    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    np.random.seed(0)

    np.random.shuffle(all_indices)
    all_processed_indices = all_indices[sample_per_process*args.index:sample_per_process*(args.index+1)]

    print(all_processed_indices)
    
    for index in tqdm(all_processed_indices, dynamic_ncols=True):
        run_blender_render(index)


if __name__ == "__main__":
    main()
