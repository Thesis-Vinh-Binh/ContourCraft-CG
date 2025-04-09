'''
0. split training & testing
1. generate rest-pose smplx-pkl
2. randomly sample upper body + lower body garment
3. quality assessment
4. choose gender, texture
5. collision handling, upper-lower garment handling
'''
import os
import sys
import numpy as np
import torch
import pickle as pkl
from tqdm import tqdm
import random
import yaml
import argparse
import trimesh
import sys
import json

sys.path.insert(0, '/is/cluster/fast/sbian/github/ContourCraft/')

from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.io import IO, save_obj, load_obj
from pytorch3d.ops import norm_laplacian, laplacian
from pytorch3d.ops import knn_gather, knn_points

from scipy.spatial.transform import Rotation
from runners.smplx.body_models import SMPLXLayer

device = torch.device("cuda")
smplx_layer = SMPLXLayer(
    '/is/cluster/fast/sbian/github/BEDLAM/data/body_models/smplx/models/smplx/SMPLX_NEUTRAL.pkl',
    ext='pkl',
    num_betas=300
).to(device=device)

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, required=True, help="path to the save resules shapenet dataset")
parser.add_argument("--wb", type=int, default=0, help="path to the save resules shapenet dataset")
args = parser.parse_args()



def calculate_pinned_v(vertice, smpl_joints, smpl_verts, lower_garment=True):
    smpl_joints = smpl_joints.reshape(-1, 3)
    # print(
    #     vertice.shape, smpl_joints.shape, smpl_verts.shape
    # )
    if lower_garment:
        smpl_root_z = smpl_joints[0, 2]
        smpl_root_x = smpl_joints[0, 0]
        pinned_vs = []
        
        angle = torch.atan2(vertice[:, 2] - smpl_root_z, vertice[:, 0] - smpl_root_x)
        for i in range(6):
            min_angle = 2 * np.pi / 6 * i - np.pi
            max_angle = 2 * np.pi / 6 * (i + 1) - np.pi

            mask = (angle > min_angle) & (angle <= max_angle)

            # if mask.sum() == 0:
            #     print('min_angle', min_angle, 'max_angle', max_angle, 'angle', angle.max(), angle.min())
            #     continue

            vertice_tmp = vertice.clone()
            vertice_tmp[~mask] = -10
            pinned_idx = torch.argmax(vertice_tmp[:, 1])
            pinned_vs.append(pinned_idx.item())

    else:
        left_shoulder = smpl_joints[16] + torch.tensor([0, 0.03, 0], device=smpl_joints.device)
        right_shoulder = smpl_joints[17] + torch.tensor([0, 0.03, 0], device=smpl_joints.device)

        closest_idx_left = torch.argmin(torch.norm(vertice - left_shoulder, dim=-1))
        closest_idx_right = torch.argmin(torch.norm(vertice - right_shoulder, dim=-1))

        pinned_vs = [closest_idx_left.item(), closest_idx_right.item()]
    
    closest_idx_on_smpl = []
    for pinned_v in pinned_vs:
        dist_v_smpl = torch.norm(vertice[pinned_v] - smpl_verts, dim=-1)
        closest_idx_on_smpl.append(
            torch.argmin(dist_v_smpl).item()
        )

    return pinned_vs, closest_idx_on_smpl



def check_garment_quality(verts_cloth, faces_cloth, verts_body, faces_body, smplx_joints, segmentation=None, is_lower_garment=False):
    garmentmesh = Meshes(verts=[verts_cloth], faces=[faces_cloth])
    garment_edges = garmentmesh.edges_packed()
    sub_valid_flag = True

    ##### EDGE LENGTH
    cloth_edges = verts_cloth[garment_edges[:, 0]] - verts_cloth[garment_edges[:, 1]]
    cloth_edges_len = torch.norm(cloth_edges, dim=-1)

    if cloth_edges_len.max() > 0.1:
        print('cloth_edges_len', cloth_edges_len)
        sub_valid_flag = False
    
    ##### PINNED VERTICES
    pinned_vs, closest_idx_on_smpl = calculate_pinned_v(verts_cloth, smplx_joints, verts_body, is_lower_garment)
    pinned_seg_partid = segmentation[closest_idx_on_smpl]

    if is_lower_garment:
        # around hips
        target_partids = [0, 1, 2, 3, 6]
    else:
        # around shoulders
        target_partids = [3, 6, 9, 12, 13, 14, 16, 17]

    partid_flag = True
    for pinnedid in pinned_seg_partid:
        if pinnedid.item() not in target_partids:
            partid_flag = False
            break
    
    if not partid_flag:
        print('partid_flag', pinned_seg_partid, is_lower_garment)
        sub_valid_flag = False

    pinned_v = verts_cloth[pinned_vs]
    closest_v = verts_body[closest_idx_on_smpl]

    pinned_dist = torch.norm(pinned_v - closest_v, dim=-1)
    if pinned_dist.max() > 0.1:
        print('pinned_dist', pinned_dist)
        sub_valid_flag = False
    
    return sub_valid_flag




def main():
    with open('/is/cluster/fast/sbian/data/blender_simulation_garmentcode_llava_labels/random_configures/summary_dict_newgarments.json', 'r') as f:
        summary_dict_newgarments = json.load(f)
    
    sampled_clothes_list = list(summary_dict_newgarments.keys())
    sampled_clothes_list = sorted(sampled_clothes_list)

    sampled_clothes_list_len = len(sampled_clothes_list)
    sample_per_process = sampled_clothes_list_len // 100 + 1
    all_indices = np.arange(sampled_clothes_list_len)

    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    np.random.seed(0)

    # all_processed_indices = all_indices[sample_per_process*args.index:sample_per_process*(args.index+1)]
    all_processed_indices = all_indices
    print(all_processed_indices)

    bm_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params.pkl'
    with open(bm_params_path, 'rb') as f:
        bm_params = pkl.load(f)
    
    bm_params = {k: torch.from_numpy(v).float().cuda() for k, v in bm_params.items() if isinstance(v, np.ndarray)} 
    bm = IO().load_mesh('/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/mean_all_1.obj', device='cuda')
    smplx_joints = bm_params['joints']
    lbs_weights = smplx_layer.lbs_weights
    segmentation = lbs_weights.argmax(dim=-1)

    CNT = 0

    for index in tqdm(all_processed_indices, dynamic_ncols=True):
        change_name = sampled_clothes_list[index]
        all_garments = summary_dict_newgarments[change_name]

        for garment in all_garments:
            if not garment.startswith('valid_garment_'):
                continue

            if args.wb:
                garment = garment + '_newwb'

            folder_name = os.path.join(
                '/is/cluster/fast/sbian/data/blender_simulation_garmentcode_llava_labels/random_configures/new_garments', 
                garment, garment
            )
            if not os.path.exists(folder_name):
                continue
            # else:
            #     print('folder_name exist', folder_name)

            paths = [item for item in os.listdir(folder_name) if item.endswith('_sim.obj')]
            if len(paths) == 0:
                print('no garment', folder_name)
                continue
                
            path = paths[0]
            design_path = os.path.join(
                '/is/cluster/fast/sbian/data/blender_simulation_garmentcode_llava_labels/random_configures/new_garments', 
                garment, 'design.yaml'
            )
            with open(design_path, 'r') as f:
                design = yaml.safe_load(f)
            
            is_lower = design['design']['meta']['upper']['v'] is None

            garment_mesh = IO().load_mesh(os.path.join(folder_name, path), include_textures=False)
            garment_verts = garment_mesh.verts_packed().to(device) * 0.01
            garment_faces = garment_mesh.faces_packed().to(device)

            if len(garment_verts) == 0 or len(garment_faces) == 0:
                print('empty garment', folder_name)
                continue

            valid_flag = check_garment_quality(
                garment_verts, garment_faces, bm.verts_packed(), bm.faces_packed(), smplx_joints, 
                segmentation=segmentation, is_lower_garment=is_lower
            )

            if not valid_flag:
                print('is_lower', is_lower,  design['design']['meta']['upper'])
                print('invalid garment', folder_name)

            with open(os.path.join(folder_name, 'quality.txt'), 'w') as f:
                f.write(str(valid_flag))
            
            # print(folder_name)

            CNT += 1

    print('CNT', CNT)

    return



if __name__ == '__main__':
    # generate_smplx_rest_pkl()
    # get_info()
    main()