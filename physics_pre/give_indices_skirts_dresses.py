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

    all_walking_motions_record = 'exp/choosed_motion_paths.json'
    with open(all_walking_motions_record, 'r') as f:
        all_walking_motions = json.load(f)

    all_used_indices = {}

    for i, sampled_clothes_dict in enumerate(tqdm(sampled_clothes_list, dynamic_ncols=True)):
        if 'upper_garment' in sampled_clothes_dict:
            lower_garment = sampled_clothes_dict['lower_garment']

            garment_type = get_type(lower_garment)
            if garment_type != 'skirt' :
                continue
            
            all_used_indices[i] = random.randint(0, len(all_walking_motions) - 1)
        
        else:
            garment = sampled_clothes_dict['whole_garment']
            garment_type = get_type(garment)

            if garment_type != 'dress' :
                continue
                
            all_used_indices[i] = random.randint(0, len(all_walking_motions) - 1)
    
    print(len(all_used_indices))
    with open('exp/all_used_indices_skirts_dresses.json', 'w') as f:
        json.dump(all_used_indices, f)