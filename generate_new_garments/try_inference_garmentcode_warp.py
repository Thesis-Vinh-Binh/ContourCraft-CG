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
parser.add_argument("--index", type=int, required=True, help="index")
args = parser.parse_args()



def run_blender_render(index):
    process = subprocess.Popen(
        ["/is/cluster/fast/sbian/anaconda3/envs/hood0/bin/python", 
        "generate_new_garments/try_inference_garmentcode.py", "--index", str(index)], stdout=subprocess.PIPE
    )

    process.wait()
    print('finished', index, process.returncode)
    return


def main():
    sampled_clothes_list_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/sampled_clothes_v4_combine_all.pkl'
    with open(sampled_clothes_list_path, 'rb') as f:
        sampled_clothes_list = pkl.load(f)

    sampled_clothes_list_len = len(sampled_clothes_list)
    sample_per_process = sampled_clothes_list_len // 100 + 1
    all_indices = np.arange(sampled_clothes_list_len)

    random.seed(2)
    os.environ['PYTHONHASHSEED'] = str(2)
    np.random.seed(2)

    np.random.shuffle(all_indices)
    all_processed_indices = all_indices[sample_per_process*args.index:sample_per_process*(args.index+1)]

    print(all_processed_indices)
    
    for index in tqdm(all_processed_indices, dynamic_ncols=True):
        run_blender_render(index)


if __name__ == "__main__":
    main()
