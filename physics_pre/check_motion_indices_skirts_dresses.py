import os

import base64
import requests
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
import yaml


default_configs = {}

def get_type(garment_path):
    garment_name = garment_path.split('/')[-1]
    garment_parent_path = '/'.join(garment_path.split('/')[:-1])
    # print('garment_parent_path', garment_parent_path)

    if garment_parent_path not in default_configs:
        config_path = os.path.join(garment_parent_path, 'dataset_properties_default_body.yaml')
        with open(config_path, 'r') as f:
            config_all = yaml.safe_load(f)
        
        default_configs[garment_parent_path] = config_all['generator']['stats']['garment_types']
    
    assert garment_parent_path in default_configs

    default_config = default_configs[garment_parent_path]
    used_config = default_config[garment_name]

    return used_config['main']



def get_all_paths_garmentcode(tmptrial=False):
    motion_dir = "/is/cluster/fast/sbian/data/bedlam_motion_for_blender/"

    if not tmptrial:
        saved_folder = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v2_nocol'
        saved_folder_final = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v2_nocol'
    
    else:
        saved_folder = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v2_tmp'
        saved_folder_final = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v2_tmp'

    sampled_clothes_list_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/sampled_clothes_motion_list_train_v2.pkl'
    with open(sampled_clothes_list_path, 'rb') as f:
        sampled_clothes_list = pkl.load(f)

    return motion_dir, saved_folder, saved_folder_final, sampled_clothes_list



if __name__ == "__main__":

    random.seed(0)
    
    device = 'cuda'
    motion_dir, saved_folder, saved_folder_final, sampled_clothes_list = get_all_paths_garmentcode(tmptrial=True)
    processed_mesh_dir = '/is/cluster/fast/sbian/data/blender_simulation_garmentcode_restpose_new'
    sampled_clothes_list = sampled_clothes_list[:10000]

    with open('exp/all_used_indices_skirts_dresses.json', 'r') as f:
        all_used_indices = json.load(f)
    
    with open('exp/choosed_motion_paths_loose.json', 'r') as f:
        choosed_motion_paths = json.load(f)

    all_used_indices2 = []
    for i, all_used_index in enumerate(tqdm(all_used_indices, dynamic_ncols=True)):
        sampled_clothes_dict = sampled_clothes_list[int(all_used_index)]
        motion_path = sampled_clothes_dict['motion_path']
        motion_path = os.path.join('/is/cluster/fast/sbian/data/bedlam_motion_for_blender/', motion_path)
        if motion_path not in choosed_motion_paths:
            print('motion_path', motion_path + ' not in choosed_motion_paths', choosed_motion_paths[i])
            continue
        all_used_indices2.append(all_used_index)

    print('all_used_indices2', len(all_used_indices2))
    all_used_indices2 = [int(item) for item in all_used_indices2]
    all_used_indices2 = sorted(all_used_indices2)
    print(all_used_indices2[:100])

    with open('exp/all_used_indices_skirts_dresses2.json', 'w') as f:
        json.dump(all_used_indices2, f)

    
   