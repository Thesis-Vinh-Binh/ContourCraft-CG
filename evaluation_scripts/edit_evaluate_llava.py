import os
import sys
import numpy as np
import torch
import pickle
import argparse
from tqdm import tqdm
import json

import subprocess

from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene
from pytorch3d.io import IO, save_obj, load_ply
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes

sys.path.insert(0, '/is/cluster/fast/sbian/github/ContourCraft/')

from utils.close_utils import get_seged_points
from utils.smplx_garment_conversion import deform_garments
from runners.smplx.body_models import SMPLXLayer


argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, required=True, help='Path to the folder containing the results')
argparser.add_argument('--method', type=str, default='llava', help='Path to the folder containing the results')
args = argparser.parse_args()


smplx_layer = SMPLXLayer(
    '/is/cluster/fast/sbian/github/BEDLAM/data/body_models/smplx/models/smplx/SMPLX_NEUTRAL.pkl',
    ext='pkl',
    num_betas=300
).cuda()

pkl_path = '/is/cluster/fast/scratch/gbecherini/siyuan/240930/smplxn_params.pkl'
with open(pkl_path, 'rb') as f:
    smplx_data = pickle.load(f)

def get_meshes_llava(path):
    # runs/try_v16_13b_lr1e_4_v3_garmentcontrol_4h100_openai_imgs_cropped_crop/vis_new/valid_garment_00170__Inner__Take1/valid_garment_lower/valid_garment_lower/valid_garment_lower_sim.obj
    llava_parent_folder = '/is/cluster/fast/sbian/github/LLaVA/'
    path = os.path.join(llava_parent_folder, path)
    args.path = path
    all_folders = os.listdir(os.path.join(path, 'vis_new'))
    all_folders = [item for item in all_folders if os.path.isdir(os.path.join(path, 'vis_new', item))]
    mesh_dict = {}
    for folder in tqdm(all_folders, dynamic_ncols=True):
        folder_name = folder[len('valid_garment_'):]
        mesh_dict[folder_name] = {}
        img_result_dir = os.path.join(path, 'vis_new', folder)
        subfolders = os.listdir(img_result_dir)
        subfolders = [item for item in subfolders if os.path.isdir(os.path.join(img_result_dir, item))]
        for subfolder in subfolders:
            garment_path = os.path.join(img_result_dir, subfolder, subfolder, f'{subfolder}_sim.obj')
            if not os.path.exists(garment_path):
                continue
            mesh = IO().load_mesh(garment_path)
            mesh_dict[folder_name][subfolder] = mesh
        
        if len(mesh_dict[folder_name]) == 0:
            mesh_dict.pop(folder_name)
            continue

        meshes_all = list(mesh_dict[folder_name].values())
        garment_combined = join_meshes_as_scene(meshes_all)
        mesh_dict[folder_name]['combined'] = garment_combined.cuda()
        mesh_dict[folder_name]['folder'] = img_result_dir

    smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params.pkl'
    with open(smplx_params_path, 'rb') as f:
        smplx_params = pickle.load(f)
    
    smplx_dict = {
        'betas': torch.tensor(smplx_params['pred_shape'], dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['pred_pose'], dtype=torch.float32).reshape(1, 165).cuda(),
        'transl': torch.tensor(smplx_params['pred_transl'], dtype=torch.float32).reshape(1, 3).cuda(),
    }
    return mesh_dict, smplx_dict


def get_mesh_gt(path):
    path = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_eva_pair/'
    saved_folder_parent = '/is/cluster/fast/sbian/github/LLaVA/runs/all_edit_results_source'
    with open('/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_eva_pair/evaluation_data.json', 'r') as f:
        evaluation_data = json.load(f)
    
    mesh_dict = {}
    for item in evaluation_data:
        target_garment = item['garment_final']
        source_garment = item['garment_init']
        print('target_garment', target_garment)
        folder_paths = target_garment.split('/')
        folder_name = folder_paths[-3] + '__' + folder_paths[-2]
        mesh_dict[folder_name] = {}

        mesh_path = [item for item in os.listdir(source_garment) if item.endswith('_sim.obj')]
        assert len(mesh_path) == 1
        mesh_path = os.path.join(source_garment, mesh_path[0])
        mesh_dict[folder_name]['combined'] = IO().load_mesh(mesh_path)

        print('folder_name', folder_name)

        saved_folder = os.path.join(saved_folder_parent, folder_name)
        os.makedirs(saved_folder, exist_ok=True)
        mesh_dict[folder_name]['folder'] = saved_folder

        print('saved_folder', saved_folder)
    
    return mesh_dict



def run_python(garmentpath, garmentpath2=None, saved_folder='', idx=0):
    if garmentpath2 is None:
        process = subprocess.Popen(
            ["/is/cluster/fast/sbian/data/blender-3.6.14-linux-x64/blender", 
            "--background", "--python", "blender_rendering_eva.py", 
            "--", "--garmentpath", garmentpath, "--savedfolder", saved_folder,
            "--texture_idx", str(idx)
            ], stdout=subprocess.PIPE
        )
    else:
        process = subprocess.Popen(
            ["/is/cluster/fast/sbian/data/blender-3.6.14-linux-x64/blender", 
            "--background", "--python", "blender_rendering_eva.py", 
            "--", "--garmentpath", garmentpath, "--garmentpath2", garmentpath2, "--savedfolder", saved_folder
            ], stdout=subprocess.PIPE
        )

    process.wait()
    print('finished', garmentpath, saved_folder, process.returncode)
    return




def fscore_func(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2




def convert_garments(pred_garment_mesh, img_name, smplx_params_raw, saved_folder=''):
    print('Start converting garments', img_name)
    # 0__valid_garment_sleeve_3
    idx, garmentname = img_name.split('__')
    target_garment_path = os.path.join(f'/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_eva_pair/{idx}/{garmentname}/{garmentname}/{garmentname}_sim.obj')

    target_mesh = IO().load_mesh(target_garment_path)
    gt_points = sample_points_from_meshes(target_mesh, 10000) * 0.01
    pred_points = sample_points_from_meshes(pred_garment_mesh, 10000) * 0.01

    print('pred_points', pred_points.shape)
    print('gt_points', gt_points.shape)

    chamfer_dist = chamfer_distance(pred_points.cuda(), gt_points.cuda())
    print(chamfer_dist)

    chamfer_x, chamfer_y = chamfer_distance(pred_points.cuda(), gt_points.cuda(), batch_reduction=None, point_reduction=None)[0]
    # print(chamfer_x.mean(), chamfer_y.mean())
    # chamfer_x = torch.sqrt(chamfer_x)
    # chamfer_y = torch.sqrt(chamfer_y)
    fscore = fscore_func(chamfer_x, chamfer_y)[0]
    print('fscore', fscore)

    return chamfer_dist[0] * 1e3, fscore


if __name__ == '__main__':
    chamfer_dist_all = []
    fscore_all = []
    # if args.method == 'llava':
    #     mesh_dict, smplx_dict = get_meshes_llava(args.path)
    #     summary_dict = {}
    #     for img_name, pred_garment_mesh_dict in mesh_dict.items():
    #         chamfer_dist, fscore = convert_garments(
    #             pred_garment_mesh_dict['combined'].cuda(), img_name, smplx_dict, saved_folder=pred_garment_mesh_dict['folder'])

    #         summary_dict[img_name] = chamfer_dist
    #         chamfer_dist_all.append(chamfer_dist)
    #         fscore_all.append(fscore)
        
    #     print('fscore_all', torch.tensor(fscore_all).mean())
    #     print('chamfer_dist_all', torch.tensor(chamfer_dist_all).mean())
    #     with open(os.path.join(args.path, 'vis_new', 'summary_dict.pkl'), 'wb') as f:
    #         pickle.dump(summary_dict, f)
    mesh_dict = get_mesh_gt(args.path)
    for img_name, pred_garment_mesh_dict in mesh_dict.items():
        mesh = pred_garment_mesh_dict['combined']
        verts = mesh.verts_packed() * 0.01
        faces = mesh.faces_packed()
        mesh = Meshes(verts=[verts], faces=[faces])

        IO().save_mesh(mesh, os.path.join(pred_garment_mesh_dict['folder'], f'{img_name}_converted.obj'))
        input_path = os.path.join(pred_garment_mesh_dict['folder'], f'{img_name}_converted.obj')

        run_python(input_path, saved_folder=pred_garment_mesh_dict['folder'], idx=3)


            