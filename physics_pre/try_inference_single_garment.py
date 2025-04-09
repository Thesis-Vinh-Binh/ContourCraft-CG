import os
import sys

sys.path.insert(0, '/is/cluster/fast/sbian/github/ContourCraft/')

HOOD_PROJECT =  "/is/cluster/fast/sbian/github/ContourCraft/"
HOOD_DATA = "/is/cluster/fast/sbian/github/HOOD/hood_data"

os.environ["HOOD_PROJECT"] = HOOD_PROJECT
os.environ["HOOD_DATA"] = HOOD_DATA

from utils.mesh_creation import obj2template
from utils.common import pickle_dump
from utils.defaults import DEFAULTS
from pathlib import Path
import random
import json

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
from utils.anypose_utils import get_rest_garemnt_info_easy, calculate_smpl_based_transform_warp_easy, calculate_pinned_v
from runners.smplx.body_models import SMPL, SMPLXLayer
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, IO, load_ply
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.ops.knn import knn_points, knn_gather
from utils.datasets import make_fromanypose_dataloader
from utils.validation import apply_material_params, apply_material2_params
from utils.common import random_between_log

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, required=True, help="path to the save resules shapenet dataset")
parser.add_argument("--seed", type=int, default=-1, help="path to the save resules shapenet dataset")
parser.add_argument("--tmptrial", action='store_true', help="path to the save resules shapenet dataset")
args = parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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


def get_all_paths_garmentcode(tmptrial=False):
    motion_dir = "/is/cluster/fast/sbian/data/bedlam_motion_for_blender/"

    if not tmptrial:
        saved_folder = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v2_nocol'
        saved_folder_final = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v2_nocol'
    
    else:
        saved_folder = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v2_physics'
        saved_folder_final = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v2_physics'

    sampled_clothes_list_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/sampled_clothes_motion_list_train_v2.pkl'
    with open(sampled_clothes_list_path, 'rb') as f:
        sampled_clothes_list = pkl.load(f)

    return motion_dir, saved_folder, saved_folder_final, sampled_clothes_list

smplx_layer = SMPLXLayer(
    '/is/cluster/fast/sbian/github/BEDLAM/data/body_models/smplx/models/smplx/SMPLX_NEUTRAL.pkl',
    ext='pkl',
    num_betas=300
).cuda()

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
    
    pinned_vertices, closest_idx_on_smpl = calculate_pinned_v(
        garment_rest_verts.reshape(-1, 3), joints_rest, rest_smpl_V, lower_garment=lower_garment
    )
    return pinned_vertices, closest_idx_on_smpl


if __name__ == "__main__":
    if args.seed > 0:
        setup_seed(args.seed * 13171 + args.index * 100000)
    else:
        setup_seed(args.index)

    config_dict_upper = get_random_params()
    config_dict_lower = get_random_params()

    skirt_path = 'exp/all_used_indices_skirts_dresses.json'
    with open(skirt_path, 'r') as f:
        all_skirt_indices_dict = json.load(f)
    
    all_skirt_indices = list(all_skirt_indices_dict.keys())
    all_skirt_indices = sorted([int(item) for item in all_skirt_indices])
    print('all_skirt_indices', len(all_skirt_indices))

    device = 'cuda'
    motion_dir, saved_folder, saved_folder_final, sampled_clothes_list = get_all_paths_garmentcode(tmptrial=args.tmptrial)
    processed_mesh_dir = '/is/cluster/fast/sbian/data/blender_simulation_garmentcode_restpose_new'
    sampled_clothes_dict = sampled_clothes_list[all_skirt_indices[args.index]]

    if args.seed > 0:
        saved_folder_i = os.path.join(saved_folder, f'{args.index}_{args.seed}', 'motion_0')
    else:
        saved_folder_i = os.path.join(saved_folder, f'{args.index}', 'motion_0')

    if not os.path.exists(saved_folder_i):
        os.makedirs(saved_folder_i)
    obj_path = os.path.join(saved_folder_i, 'combined_garment.obj')
    # if os.path.exists(obj_path):
    #     assert False

    if 'upper_garment' in sampled_clothes_dict:
        lower_garment = sampled_clothes_dict['lower_garment']
        lower_name = lower_garment.split('/')[-1]

        motion_path = sampled_clothes_dict['motion_path']
        motion_path = os.path.join(motion_dir, motion_path)
        lower_garment_obj = os.path.join(lower_garment, f'{lower_name}_sim_connected.obj')

        body_measurements_path = os.path.join(sampled_clothes_dict['lower_garment'], f'{lower_name}_body_measurements.yaml')
        with open(body_measurements_path, 'r') as f:
            body_measurements = yaml.load(f, Loader=yaml.FullLoader)
            body_name = body_measurements['body']['body_sample']
        
        if body_name == 'mean_all_apart':
            motion_path = motion_path.replace('_300.npz', '_300_apart.npz')
            rest_smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params_apart.pkl'
        else:
            rest_smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params.pkl'
        
        with open(rest_smplx_params_path, 'rb') as f:
            rest_smplx_params = pkl.load(f)

        obj_path = os.path.join(saved_folder_i, 'combined_garment.obj')
        mesh_lower = trimesh.load(lower_garment_obj, process=True)

        mesh_lower_vertices = mesh_lower.vertices
        pinned_verts_low, closest_idx_low = get_pinned_verts(mesh_lower_vertices, rest_smplx_params, 'cuda', lower_garment=True)

        mesh_lower_faces = mesh_lower.faces

        combined_vertices = mesh_lower_vertices
        combined_faces = mesh_lower_faces
        
        combined_mesh = trimesh.Trimesh(vertices=combined_vertices * 0.01, faces=combined_faces)
        combined_mesh.export(obj_path)

        pinned_verts_final = pinned_verts_low
        closest_idx_final = closest_idx_low

    else:
        garment = sampled_clothes_dict['whole_garment']
        name = garment.split('/')[-1]
        
        motion_path = sampled_clothes_dict['motion_path']
        motion_path = os.path.join(motion_dir, motion_path)

        print('motion_path', motion_path)
        assert False

        garment = os.path.join(garment, f'{name}_sim_connected.obj')
        body_measurements_path = os.path.join(sampled_clothes_dict['whole_garment'], f'{name}_body_measurements.yaml')
        with open(body_measurements_path, 'r') as f:
            body_measurements = yaml.load(f, Loader=yaml.FullLoader)
            body_name = body_measurements['body']['body_sample']

        if body_name == 'mean_all_apart':
            motion_path = motion_path.replace('_300.npz', '_300_apart.npz')
            rest_smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params_apart.pkl'
        else:
            rest_smplx_params_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/registered_params.pkl'

        with open(rest_smplx_params_path, 'rb') as f:
            rest_smplx_params = pkl.load(f)
        
        obj_path = os.path.join(saved_folder_i, 'combined_garment.obj')
        mesh = trimesh.load(garment, process=True)
        mesh_vertices = mesh.vertices
        mesh_faces = mesh.faces

        pinned_verts_final, closest_idx_final = get_pinned_verts(mesh_vertices, rest_smplx_params, 'cuda')
        mesh = trimesh.Trimesh(vertices=mesh_vertices * 0.01, faces=mesh_faces)
        mesh.export(obj_path)
        combined_faces = mesh_faces
        combined_vertices = mesh_vertices


    # out_template_path = os.path.join(saved_folder_i, 'combined_garment.pkl')
    out_template_path = os.path.join(saved_folder, f'{args.index}', 'motion_0', 'combined_garment.pkl')
    if not os.path.exists(out_template_path):
        template_dict = obj2template(obj_path, verbose=True)
        pickle_dump(template_dict, out_template_path)
        # add_pinned_verts_single_template(out_template_path, pinned_verts_final)

    # change mesh_upper_faces + mesh_upper_vertices
    with open(out_template_path, 'rb') as f:
        template_dict = pkl.load(f)
    
    faces_new = template_dict['faces']
    assert len(faces_new) == len(combined_faces)
    verts_new = template_dict['rest_pos'].reshape(1, -1, 3)
    idx_up_max = -1

    pinned_verts_old = combined_vertices[pinned_verts_final] * 0.01
    _, idx_pinned, _ = knn_points(
        torch.from_numpy(pinned_verts_old).to(device).float().reshape(1, -1, 3), 
        torch.from_numpy(verts_new).to(device).float(), K=1
    )
    idx_pinned = idx_pinned.reshape(-1).cpu().numpy().tolist()
    add_pinned_verts_single_template(out_template_path, idx_pinned)
    print('pinned_verts_final', pinned_verts_final, idx_pinned)
    ################################################################################



    # use ContourCraft
    modules, config = load_params('aux/from_any_pose')
    checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'contourcraft.pth'

    config = apply_material_params(config, config_dict_upper)
    config = apply_material2_params(config, config_dict_lower)
    runner_name = list(config.runner.keys())[0]

    config.runner[runner_name].material2.use_meterial2 = False # use this when 2 garments
    config.runner[runner_name].material2.start_face_indices = -1
    config.runner[runner_name].material2.start_vertex_indices = -1

    runner_module, runner, aux_modules = create_runner(modules, config)
    # assert False

    state_dict =  torch.load(checkpoint_path)
    runner.load_state_dict(state_dict['training_module'])
    runner = runner.to(device)

    motion_dict = np.load(motion_path)
    pose_sequence_path = os.path.join(saved_folder_i, 'new_obstacle.pkl')
    smplx_out_final = smplx_layer.forward_simple(
        betas=torch.from_numpy(motion_dict['betas']).to(device).float().expand(len(motion_dict['poses']), -1),
        full_pose=torch.from_numpy(motion_dict['poses']).to(device).float(),
        transl=torch.from_numpy(motion_dict['trans']).to(device).float(), # zero translation
        pose2rot=True
    )

    smplx_verts = smplx_out_final.vertices.cpu().numpy()
    smplx_faces = smplx_layer.faces_tensor.cpu().numpy()

    saved_dict = {
        'verts': smplx_verts,
        'faces': smplx_faces
    }
    with open(pose_sequence_path, 'wb') as f:
        pkl.dump(saved_dict, f)

    pose_sequence_type = 'mesh'
    garment_template_path = out_template_path

    dataloader = make_fromanypose_dataloader(pose_sequence_type=pose_sequence_type, 
                        pose_sequence_path=pose_sequence_path, 
                        garment_template_path=garment_template_path)


    sample = next(iter(dataloader))

    trajectories_dict = runner.valid_rollout(sample)

    # Save the sequence to disk
    out_path = os.path.join(saved_folder_i, 'output_anypose.pkl')
    print(f"Rollout saved into {out_path}")
    pickle_dump(dict(trajectories_dict), out_path)

    smplx_verts = smplx_out_final.vertices.cpu()
    smplx_faces = smplx_layer.faces_tensor.cpu().numpy()
    
    pred_pos = trajectories_dict['pred']
    cloth_faces = trajectories_dict['cloth_faces']
    print('cloth_faces', cloth_faces.shape)

    pred_pos = torch.tensor(pred_pos).float()
    cloth_faces = torch.tensor(cloth_faces).long()

    metrics = trajectories_dict['metrics']
    print('metrics', list(metrics.keys()))

    saved_mesh_dir = os.path.join(saved_folder_i, 'mesh_tmp')
    if not os.path.exists(saved_mesh_dir):
        os.makedirs(saved_mesh_dir)

    seq_len = len(pred_pos) - 2
    sampled_indices = np.arange(0, seq_len, 25)
    for frame_number in sampled_indices:
        smplx_mesh = Meshes([smplx_verts[frame_number]], [torch.tensor(smplx_faces).long()])
        garment_mesh = Meshes([pred_pos[frame_number+2]], [cloth_faces])

        combined_mesh = join_meshes_as_scene([smplx_mesh, garment_mesh])

        IO().save_mesh(
            combined_mesh, os.path.join(saved_mesh_dir, f'{frame_number}_0.obj')
        )

        print(os.path.join(saved_mesh_dir, f'{frame_number}_0.obj'))

    # pred_pos = trajectories_dict['pred'] - obstacle_pos_mean.reshape(-1, 1, 3) + smplx_verts_mean[2:].reshape(-1, 1, 3)
    pred_pos = trajectories_dict['pred']
    trajectories_dict_saved = dict(
        pred=pred_pos,
        cloth_faces=trajectories_dict['cloth_faces'],
        pinned_vertices=pinned_verts_final,
        closest_idx_on_smpl=closest_idx_final,
        metrics=metrics,
        config_dict_lower=config_dict_lower,
        config_dict_upper=config_dict_upper
    )
    np.savez(out_path.replace('.pkl', '.npz'), **trajectories_dict_saved)

