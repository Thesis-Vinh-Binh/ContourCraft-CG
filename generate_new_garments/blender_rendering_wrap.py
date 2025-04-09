import argparse
import time
import subprocess
import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, required=True, help="path to the save resules shapenet dataset")
args = parser.parse_args()

def run_python(index, frameid):

    process = subprocess.Popen(
        ["/is/cluster/fast/sbian/data/blender-3.6.14-linux-x64/blender", 
        "--background", "--python", "generate_new_garments/blender_rendering.py", 
        "--", "--index", str(index), "--frameid", str(frameid)
        ], stdout=subprocess.PIPE
    )

    process.wait()
    print('finished', index, process.returncode)
    return

def main():
    saved_folder_final = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v4'
    sampled_clothes_list_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/sampled_clothes_v4_combine_all.pkl'
    with open(sampled_clothes_list_path, 'rb') as f:
        sampled_clothes_list = pkl.load(f)

    
    sampled_clothes_list_len = len(sampled_clothes_list)
    all_indices = np.arange(sampled_clothes_list_len)
    sample_per_process = sampled_clothes_list_len // 100 + 1

    random.seed(2)
    os.environ['PYTHONHASHSEED'] = str(2)
    np.random.seed(2)

    np.random.shuffle(all_indices)
    all_processed_indices = all_indices[sample_per_process*args.index:sample_per_process*(args.index+1)]

    for index in tqdm(all_processed_indices, dynamic_ncols=True):
        obj_dir = os.path.join(
            saved_folder_final, f'{index}', f'motion_0/meshes/', 
        )
        if not os.path.exists(obj_dir):
            print('obj_dir not exist', obj_dir)
            continue
        
        suffix = '.obj'
        body_paths = [item for item in os.listdir(obj_dir) if item.startswith('SMPLX-mesh-neutral') and item.endswith(suffix)]
        for body_path in body_paths:
            frameid = int(body_path.split('_')[-1].replace(suffix, ''))
            print(index, frameid)
            run_python(index, frameid)



if __name__ == "__main__":
    main()
