import os
import sys

HOOD_PROJECT =  "/is/cluster/fast/sbian/github/ContourCraft/"
HOOD_DATA = "/is/cluster/fast/sbian/github/HOOD/hood_data"

os.environ["HOOD_PROJECT"] = HOOD_PROJECT
os.environ["HOOD_DATA"] = HOOD_DATA

sys.path.insert(0, HOOD_PROJECT)

from utils.mesh_creation import obj2template
from utils.common import pickle_dump
from utils.defaults import DEFAULTS
from pathlib import Path
import random

from utils.arguments import load_params, create_modules
from utils.common import move2device
from utils.io import pickle_dump
from utils.defaults import DEFAULTS
from pathlib import Path
import torch
from utils.arguments import create_runner
import pickle as pkl
from utils.mesh_creation import add_pinned_verts_single_template
import argparse
import yaml
import trimesh
import numpy as np
from utils.anypose_utils import get_rest_garemnt_info_easy, calculate_smpl_based_transform_warp_easy, calculate_pinned_v_dense
from runners.smplx.body_models import SMPL, SMPLXLayer
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, IO, load_ply
from pytorch3d.structures.meshes import join_meshes_as_scene
from utils.datasets import make_fromanypose_dataloader
from utils.validation import apply_material_params, apply_material2_params
from utils.common import random_between_log

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, required=True, help="path to the save resules shapenet dataset")
args = parser.parse_args()

device = 'cpu'

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_random_params():
    lame_mu_min = 15909
    lame_mu_max = 63636
    lame_lambda_min = 3535.414406069427
    lame_lambda_max = 93333.73508005822
    bending_coeff_min = 6.370782056371576e-08
    bending_coeff_max = 0.0013139737991266374
    density_min = 4.34e-2
    density_max = 7e-1

    config_dict = dict()
    size = [1]
    config_dict['density'] = random_between_log(density_min, density_max, size).item()
    config_dict['lame_mu'] = random_between_log(lame_mu_min, lame_mu_max, size).item()
    config_dict['lame_lambda'] = random_between_log(lame_lambda_min, lame_lambda_max, size).item()
    config_dict['bending_coeff'] = random_between_log(bending_coeff_min, bending_coeff_max, size).item()
    config_dict['smpl_model'] = None

    return config_dict


def get_all_paths_garmentcode():
    motion_dir = "/is/cluster/fast/sbian/data/bedlam_motion_for_blender/"
    saved_folder = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v4'

    saved_folder_final = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v4'
    sampled_clothes_list_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/sampled_clothes_v4_combine_all.pkl'
    with open(sampled_clothes_list_path, 'rb') as f:
        sampled_clothes_list = pkl.load(f)

    return motion_dir, saved_folder, saved_folder_final, sampled_clothes_list


smplx_layer = SMPLXLayer(
    '/is/cluster/fast/sbian/github/BEDLAM/data/body_models/smplx/models/smplx/SMPLX_NEUTRAL.pkl',
    ext='pkl',
    num_betas=300
).to(device)


def get_pinned_verts(garment_rest_verts, rest_smplx_params, device, lower_garment=False):
    garment_rest_verts = torch.from_numpy(garment_rest_verts).to(device).float().unsqueeze(0) * 0.01

    ##### calculate the smpl-based vertices
    rest_A, rest_smpl_V, smpl_based_lbs_weight = get_rest_garemnt_info_easy(
        rest_smplx_params, smplx_layer, garment_rest_verts
    )

    joints_rest = torch.from_numpy(rest_smplx_params['joints']).float().reshape(-1, 3).to(device)
    if 'smplx_vertices' in rest_smplx_params:
        rest_smpl_V = torch.from_numpy(rest_smplx_params['smplx_vertices']).float().reshape(-1, 3).to(device)
    else:
        rest_smpl_V = torch.from_numpy(rest_smplx_params['vertices']).float().reshape(-1, 3).to(device)
    
    pinned_vertices, closest_idx_on_smpl = calculate_pinned_v_dense(
        garment_rest_verts.reshape(-1, 3), None, joints_rest, rest_smpl_V, lower_garment=lower_garment
    )

    return pinned_vertices



if __name__ == "__main__":
    setup_seed(args.index)

    motion_dir, saved_folder, saved_folder_final, sampled_clothes_list = get_all_paths_garmentcode()
    indices = [i for i in range(len(sampled_clothes_list))]
    # random.shuffle(indices)
    args.index = indices[args.index]

    sampled_clothes_dict = sampled_clothes_list[args.index]

    if True:
        saved_folder_i = os.path.join(saved_folder, f'{args.index}', 'motion_0')
        if not os.path.exists(saved_folder_i):
            os.makedirs(saved_folder_i)

        obj_path = os.path.join(saved_folder_i, 'combined_garment.obj')

    if 'upper_garment' in sampled_clothes_dict:
        upper_garment = sampled_clothes_dict['upper_garment']
        lower_garment = sampled_clothes_dict['lower_garment']

        upper_name = upper_garment.split('/')[-1]
        lower_name = lower_garment.split('/')[-1]

        motion_path = sampled_clothes_dict['motion_path']
        motion_path = os.path.join(motion_dir, motion_path)

        # if 'random_configures/new_garments' not in upper_garment:
        if os.path.exists(os.path.join(upper_garment, f'{upper_name}_sim_connected.obj')):
            upper_garmen_obj = os.path.join(upper_garment, f'{upper_name}_sim_connected.obj')
            lower_garment_obj = os.path.join(lower_garment, f'{lower_name}_sim_connected.obj')
        else:
            upper_garmen_obj = os.path.join(upper_garment, f'{upper_name}_sim.obj')
            lower_garment_obj = os.path.join(lower_garment, f'{lower_name}_sim.obj')

        body_name = sampled_clothes_dict['body_name']
        
        if body_name == 'mean_all_apart':
            motion_path = motion_path.replace('_300.npz', '_300_apart.npz')
            rest_smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params_apart.pkl'
        else:
            rest_smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params.pkl'
        
        with open(rest_smplx_params_path, 'rb') as f:
            rest_smplx_params = pkl.load(f)

        mesh_upper = IO().load_mesh(upper_garmen_obj, include_textures=False)
        mesh_lower = IO().load_mesh(lower_garment_obj, include_textures=False)

        mesh_upper  = trimesh.Trimesh(vertices=mesh_upper.verts_packed().numpy(), faces=mesh_upper.faces_packed().numpy(), process=True)
        mesh_lower  = trimesh.Trimesh(vertices=mesh_lower.verts_packed().numpy(), faces=mesh_lower.faces_packed().numpy(), process=True)

        mesh_upper_vertices = mesh_upper.vertices
        mesh_lower_vertices = mesh_lower.vertices

        pinned_verts_up = get_pinned_verts(mesh_upper_vertices, rest_smplx_params, device)
        pinned_verts_low = get_pinned_verts(mesh_lower_vertices, rest_smplx_params, device, lower_garment=True)

        mesh_upper_faces = mesh_upper.faces
        mesh_lower_faces = mesh_lower.faces + len(mesh_upper_vertices)

        combined_vertices = np.concatenate([mesh_upper_vertices, mesh_lower_vertices], axis=0)
        combined_faces = np.concatenate([mesh_upper_faces, mesh_lower_faces], axis=0)
        
        combined_mesh = trimesh.Trimesh(vertices=combined_vertices * 0.01, faces=combined_faces)
        combined_mesh.export(obj_path)

        pinned_verts_low = [item + len(mesh_upper_vertices) for item in pinned_verts_low]
        pinned_verts_final = pinned_verts_up + pinned_verts_low

    else:
        garment = sampled_clothes_dict['whole_garment']

        name = garment.split('/')[-1]
        motion_path = sampled_clothes_dict['motion_path']
        motion_path = os.path.join(motion_dir, motion_path)

        # if 'random_configures/new_garments' not in garment:
        if os.path.exists(os.path.join(garment, f'{name}_sim_connected.obj')):
            garment = os.path.join(garment, f'{name}_sim_connected.obj')
        else:
            garment = os.path.join(garment, f'{name}_sim.obj')

        body_name = sampled_clothes_dict['body_name']

        if body_name == 'mean_all_apart':
            motion_path = motion_path.replace('_300.npz', '_300_apart.npz')
            rest_smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params_apart.pkl'
        else:
            rest_smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params.pkl'

        with open(rest_smplx_params_path, 'rb') as f:
            rest_smplx_params = pkl.load(f)

        # mesh = trimesh.load(garment, process=True)
        mesh = IO().load_mesh(garment, include_textures=False)
        mesh = trimesh.Trimesh(vertices=mesh.verts_packed().numpy(), faces=mesh.faces_packed().numpy(), process=True)
        mesh_vertices = mesh.vertices
        mesh_faces = mesh.faces

        pinned_verts_final = get_pinned_verts(mesh_vertices, rest_smplx_params, device)
        mesh = trimesh.Trimesh(vertices=mesh_vertices * 0.01, faces=mesh_faces)
        mesh.export(obj_path)
        combined_mesh = mesh

    out_template_path = obj_path.replace('.obj', '.pkl')
    if os.path.exists(out_template_path):
        assert False

    print('obj_path', obj_path)
    template_dict = obj2template(obj_path, verbose=True, approximate_center=True)
    pickle_dump(template_dict, out_template_path)
    add_pinned_verts_single_template(out_template_path, pinned_verts_final)


