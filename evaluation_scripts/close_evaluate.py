import os
import sys
import re
import json
import numpy as np
import torch
import pickle
import argparse
from tqdm import tqdm

from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene
from pytorch3d.io import IO, save_obj, load_ply
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
import copy
from scipy.spatial.transform import Rotation as R


sys.path.insert(0, '/workspace/ContourCraft-CG/')

from utils.close_utils import get_seged_points
from utils.smplx_garment_conversion import deform_garments
from runners.smplx.body_models import SMPLXLayer
import subprocess

def argument_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--folder', '-f', type=str, required=True, help='evaluation folder')
    argparser.add_argument('--method', '-m', type=str, default='llava', help='method name')
    argparser.add_argument('--metrics', '-mt', type=str, nargs='+', default=['chamfer', 'fscore'], help='metrics to evaluate')
    argparser.add_argument('--use_cache', type=bool, default=False, help='use cache')
    args = argparser.parse_args()
    return args

smplx_layer = SMPLXLayer(
    'ccraft_data/aux_data/body_models/models/smplx/SMPLX_NEUTRAL.pkl',
    ext='pkl',
    num_betas=300
).cuda()

PKL_PATH = '/is/cluster/fast/scratch/gbecherini/siyuan/240930/smplxn_params.pkl'
BLENDER_PATH = '/workspace/blender-3.6.14-linux-x64/blender'
CLOSE_DATA_PATH = '/workspace/CloSe/assets/close-di'
EVALUATION_DATA_PATH = '/workspace/ContourCraft-CG/evaluation'

# with open(PKL_PATH, 'rb') as f:
#     smplx_data = pickle.load(f)


def rotate_pose(pose, angle, which_axis='x'):
    # pose: (n, 72)
    
    is_tensor = torch.is_tensor(pose)
    if is_tensor:
        pose = pose.cpu().detach().numpy()
    
    swap_rotation = R.from_euler(which_axis, [angle/np.pi*180], degrees=True)
    root_rot = R.from_rotvec(pose[:, :3])
    pose[:, :3] = (swap_rotation * root_rot).as_rotvec()

    if is_tensor:
        pose = torch.FloatTensor(pose)

    return pose


# runs/try_lr1e_4_wholebody_pose_v2_detailT2_upd_possibleDebug_v3_garmentcontrol_addFTdata_onlyimg_CloSE_eva_crop
def get_meshes_llava(path):
    # runs/try_v16_13b_lr1e_4_v3_garmentcontrol_4h100_openai_imgs_cropped_crop/vis_new/valid_garment_00170__Inner__Take1/valid_garment_lower/valid_garment_lower/valid_garment_lower_sim.obj
    path = os.path.join(EVALUATION_DATA_PATH, path)
    args.folder = path
    all_folders = os.listdir(os.path.join(path, 'vis_new'))
    all_folders = [item for item in all_folders if os.path.isdir(os.path.join(path, 'vis_new', item))]
    all_folders = all_folders
    mesh_dict = {}
    path_dict = {}
    for folder in tqdm(all_folders, dynamic_ncols=True): 
        folder_name = folder[len('valid_garment_'):]
        mesh_dict[folder_name] = {}
        path_dict[folder_name] = {}
        img_result_dir = os.path.join(path, 'vis_new', folder)
        subfolders = os.listdir(img_result_dir)
        subfolders = [item for item in subfolders if os.path.isdir(os.path.join(img_result_dir, item))]
        for subfolder in subfolders:
            garment_path = os.path.join(img_result_dir, subfolder, subfolder, f'{subfolder}_sim.obj')
            if not os.path.exists(garment_path):
                continue
            mesh = IO().load_mesh(garment_path, load_textures=False)
            mesh_dict[folder_name][subfolder] = mesh
            path_dict[folder_name][subfolder] = garment_path
        
        if len(mesh_dict[folder_name]) == 0:
            mesh_dict.pop(folder_name)
            path_dict.pop(folder_name)
            continue

        meshes_all = list(mesh_dict[folder_name].values())
        garment_combined = join_meshes_as_scene(meshes_all)
        # if garment_combined.verts_padded().max() > 1e3:
        #     continue
        mesh_dict[folder_name]['combined'] = garment_combined.cuda()
        mesh_dict[folder_name]['folder'] = img_result_dir

        # print(garment_combined.verts_packed().shape)

    smplx_params_path = 'assets/aaa_mesh_registrarion/registered_params.pkl'
    with open(smplx_params_path, 'rb') as f:
        smplx_params = pickle.load(f)
    
    smplx_dict = {
        'betas': torch.tensor(smplx_params['pred_shape'], dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['pred_pose'], dtype=torch.float32).reshape(1, 165).cuda(),
        'transl': torch.tensor(smplx_params['pred_transl'], dtype=torch.float32).reshape(1, 3).cuda(),
    }
    return mesh_dict, path_dict, smplx_dict

def get_meshes_d2g(path):
    path = os.path.join(EVALUATION_DATA_PATH, path)
    args.folder = path
    all_folders = os.listdir(path)
    all_folders = [item for item in all_folders if os.path.isdir(os.path.join(path, item))]
    all_folders = all_folders
    mesh_dict = {}
    path_dict = {}
    for folder in tqdm(all_folders, dynamic_ncols=True): 
        folder_name = folder
        mesh_dict[folder_name] = {}
        path_dict[folder_name] = {}
        img_result_dir = os.path.join(path, folder)
        # subfolder is the first folder in the img_result_dir
        subfolder = os.listdir(img_result_dir)[0]
        garment_path = os.path.join(img_result_dir, subfolder, f'{subfolder}_sim.obj')
        mesh = IO().load_mesh(garment_path, load_textures=False)
        mesh_dict[folder_name]['combined'] = mesh.cuda()
        mesh_dict[folder_name]['folder'] = img_result_dir

    smplx_params_path = 'assets/aaa_mesh_registrarion/registered_params.pkl'
    with open(smplx_params_path, 'rb') as f:
        smplx_params = pickle.load(f)
    
    smplx_dict = {
        'betas': torch.tensor(smplx_params['pred_shape'], dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['pred_pose'], dtype=torch.float32).reshape(1, 165).cuda(),
        'transl': torch.tensor(smplx_params['pred_transl'], dtype=torch.float32).reshape(1, 3).cuda(),
    }
    return mesh_dict, path_dict, smplx_dict


def convert_smpl_to_smplx(smpl_betas, smpl_pose, smpl_trans):
    # Betas: pad to 16 if needed
    if smpl_betas.shape[0] < 16:
        smplx_betas = np.pad(smpl_betas, (0, 16 - smpl_betas.shape[0]), 'constant')
    else:
        smplx_betas = smpl_betas

    # Pose conversion
    global_orient = smpl_pose[:3]                  # (3,)
    body_pose = smpl_pose[3:66]                    # (63,) → 21 joints

    # Build full 165-D SMPL-X pose vector
    full_pose = np.concatenate([
        global_orient,                              # (3,)
        body_pose,                                  # (63,)
        np.zeros(45),                               # left_hand_pose (15 joints × 3)
        np.zeros(45),                               # right_hand_pose (15 joints × 3)
        np.zeros(3),                                # jaw_pose
        np.zeros(6)                                 # eye_pose (2 eyes × 3)
    ])                                              # => total (165,)

    smplx_params = {
        'betas': smplx_betas,
        'poses': full_pose,
        'expression': np.zeros(10),                 # expression blendshapes
        'trans': smpl_trans
    }

    return smplx_params


def convert_garments(pred_garment_mesh, img_name, smplx_params_raw, saved_folder=''):
    print('Start converting garments', img_name)
    img_name = img_name.split('.')[0]
    garnment_id = img_name
    target_npz_path = os.path.join(
        CLOSE_DATA_PATH, f'{garnment_id}.npz'
    )
    target_npz = np.load(target_npz_path)

    smplx_params = convert_smpl_to_smplx(target_npz['betas'], target_npz['pose'], target_npz['trans'])
    gt_points_upper, gt_points_lower, gt_points_wholebody, gt_points = get_seged_points(target_npz)
    gt_points = gt_points / target_npz['scale']
    print('scale', target_npz['scale'])
    # gt_points_wholebody = torch.from_numpy(gt_points_wholebody).float().cuda()
    # print('gt_points_wholebody', gt_points_wholebody.shape)
    gt_points = torch.from_numpy(gt_points)[::10].unsqueeze(0).float().cuda()
    print('gt_points', gt_points.shape)

    betas = np.zeros(300)
    betas[:16] = smplx_params['betas']

    # print('smplx_params', list(smplx_params.keys()))

    smplx_params_new = {
        'betas': torch.tensor(betas, dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['poses'], dtype=torch.float32).reshape(1, 55, 3).cuda(), 
        'transl': torch.tensor(smplx_params['trans'], dtype=torch.float32).reshape(1, 3).cuda(),
    }

    deformed_garment_verts = deform_garments(
        smplx_layer, smplx_params_raw, smplx_params_new, pred_garment_mesh, smplx_layer.lbs_weights
    )

    deformed_garment_mesh = Meshes(verts=[deformed_garment_verts], faces=[pred_garment_mesh.faces_packed()])
    pred_points = sample_points_from_meshes(deformed_garment_mesh, len(gt_points[0]))
    # gt_points = gt_points_wholebody.unsqueeze(0)

    print('pred_points', pred_points.shape, pred_points.mean(dim=1))
    print('gt_points', gt_points.shape, gt_points.mean(dim=1))

    chamfer_dist = chamfer_distance(pred_points, gt_points)
    print(chamfer_dist)

    IO().save_mesh(deformed_garment_mesh, os.path.join(saved_folder, f'{img_name}_converted.obj'))
    pointcloud = Pointclouds(points=[gt_points[0]])
    IO().save_pointcloud(pointcloud, os.path.join(saved_folder, f'{img_name}_converted.ply'))
    print('saved_folder', saved_folder)

    return chamfer_dist[0] * 1e3



def fscore_func(dist1, dist2, threshold=0.01):
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


def calculate_fscore(pred_garment_mesh, img_name, smplx_params_raw, saved_folder=''):
    img_name = img_name.split('.')[0]
    garnment_id = img_name
    target_npz_path = os.path.join(
        CLOSE_DATA_PATH, f'{garnment_id}.npz'
    )
    target_npz = np.load(target_npz_path)

    gt_points_upper, gt_points_lower, gt_points_wholebody, gt_points = get_seged_points(target_npz)
    gt_points = gt_points / target_npz['scale']
    gt_points = torch.from_numpy(gt_points)[::10].unsqueeze(0).float().cuda()

    deformed_garment_mesh = IO().load_mesh(os.path.join(saved_folder, f'{img_name}_converted.obj'))
    pred_points = sample_points_from_meshes(deformed_garment_mesh.cuda(), len(gt_points[0]))

    # print('pred_points', pred_points.shape, pred_points.max())
    # print('gt_points', gt_points.shape, gt_points.max())

    chamfer_x, chamfer_y = chamfer_distance(pred_points, gt_points, batch_reduction=None, point_reduction=None)[0]
    # print(chamfer_x.mean(), chamfer_y.mean())
    # chamfer_x = torch.sqrt(chamfer_x)
    # chamfer_y = torch.sqrt(chamfer_y)
    # print('chamfer_x', chamfer_x.mean(), chamfer_y.mean())
    fscore = fscore_func(chamfer_x, chamfer_y)[0]
    print('fscore', fscore)

    return fscore


def run_python(garmentpath, garmentpath2=None, saved_folder=''):
    if garmentpath2 is None:
        process = subprocess.Popen(
            [BLENDER_PATH, 
            "--background", "--python", "blender_rendering_eva.py", 
            "--", "--garmentpath", garmentpath, "--savedfolder", saved_folder
            ], stdout=subprocess.PIPE
        )
    else:
        process = subprocess.Popen(
            [BLENDER_PATH, 
            "--background", "--python", "blender_rendering_eva.py", 
            "--", "--garmentpath", garmentpath, "--garmentpath2", garmentpath2, "--savedfolder", saved_folder
            ], stdout=subprocess.PIPE
        )

    process.wait()
    print('finished', garmentpath, process.returncode)
    return


def convert_garments_Apose(pred_garment_mesh, img_name, smplx_params_raw, inp_path):
    print('Start converting garments', img_name)
    smplx_params_path = 'assets/aaa_mesh_registrarion/registered_params.pkl'
    with open(smplx_params_path, 'rb') as f:
        smplx_params = pickle.load(f)
    
    smplx_params_new = {
        'betas': torch.tensor(smplx_params['pred_shape'], dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['pred_pose'], dtype=torch.float32).reshape(1, 165).cuda(),
        'transl': torch.tensor(smplx_params['pred_transl'], dtype=torch.float32).reshape(1, 3).cuda(),
    }

    deformed_garment_verts = deform_garments(
        smplx_layer, smplx_params_raw, smplx_params_new, pred_garment_mesh, smplx_layer.lbs_weights
    )

    deformed_garment_mesh = Meshes(verts=[deformed_garment_verts], faces=[pred_garment_mesh.faces_packed()])
    saved_path = inp_path.replace('.obj', '_converted_Apose.obj')
    IO().save_mesh(deformed_garment_mesh, saved_path)
    print('saved_path', saved_path)

    return saved_path

def export_to_pkl(summary_dict, output_path, file_name):
    with open(os.path.join(output_path, f'{file_name}.pkl'), 'wb') as f:
        pickle.dump(summary_dict, f)

def calculate_and_return_cd(mesh_dict, smplx_dict, folder_name):
    summary_dict = {}
    chamfer_dist_all = []
    failed = []
    for img_name, pred_garment_mesh_dict in mesh_dict.items():
        if 'smplx' in pred_garment_mesh_dict:
            smplx_dict = pred_garment_mesh_dict['smplx']
        chamfer_dist = convert_garments(
            pred_garment_mesh_dict['combined'].cuda(), img_name, smplx_dict, saved_folder=pred_garment_mesh_dict['folder'])

        if chamfer_dist > 200:
            failed.append(img_name)
            continue

        summary_dict[img_name] = chamfer_dist
        chamfer_dist_all.append(chamfer_dist)
        
    export_to_pkl(summary_dict, folder_name, 'summary_dict')
    return chamfer_dist_all, summary_dict, failed

def calculate_and_return_fscore(mesh_dict, smplx_dict, folder_name, summary_dict = None):
    fscore_dict = {}
    fscore_dist_all = []
    if summary_dict is None:
        with open(os.path.join(folder_name, 'summary_dict.pkl'), 'rb') as f:
            summary_dict = pickle.load(f)
    for img_name, pred_garment_mesh_dict in mesh_dict.items():
        if img_name not in summary_dict:
            continue
        fscore0 = calculate_fscore(
            pred_garment_mesh_dict['combined'].cuda(), img_name, smplx_dict, saved_folder=pred_garment_mesh_dict['folder'])
        fscore_dict[img_name] = fscore0
        fscore_dist_all.append(fscore0)
    export_to_pkl(fscore_dict, folder_name, 'fscore_dict')
    return fscore_dist_all, fscore_dict

def get_meshes(folder, method):
    if method == 'llava':
        return get_meshes_llava(folder)
    elif method == 'd2g':
        return get_meshes_d2g(folder)
    else:
        raise ValueError(f'Method {method} not supported')

if __name__ == '__main__':
    args = argument_parser()
    output = {}
    mesh_dict = None 
    print('args values', args.folder, args.method, args.metrics, args.use_cache)
    if not args.use_cache:
        print('not using cache')
        mesh_dict, path_dict, smplx_dict = get_meshes(args.folder, args.method)
        
    directory = os.path.join(EVALUATION_DATA_PATH, args.folder)

    if 'chamfer' in args.metrics:
        if os.path.exists(os.path.join(directory, 'summary_dict.pkl')) and args.use_cache:
            with open(os.path.join(directory, 'summary_dict.pkl'), 'rb') as f: 
                summary_dict = pickle.load(f)
                chamfer_dist_all = list(summary_dict.values())
        else:
            if mesh_dict is None:
                print('prepare for chamfer')
                mesh_dict, path_dict, smplx_dict = get_meshes(args.folder, args.method)
            chamfer_dist_all, summary_dict, failed = calculate_and_return_cd(mesh_dict, smplx_dict, directory)
            
        output['chamfer'] = {
            'mean': torch.tensor(chamfer_dist_all).mean().item(),
            'std': torch.tensor(chamfer_dist_all).std().item(),
            'median': torch.tensor(chamfer_dist_all).median().item(),
            'min': torch.tensor(chamfer_dist_all).min().item(),
            'max': torch.tensor(chamfer_dist_all).max().item(),
            'failed_count': len(failed),
        }
        output['failed'] = failed
    
    if 'fscore' in args.metrics:
        if os.path.exists(os.path.join(directory, 'fscore_dict.pkl')) and args.use_cache:
            with open(os.path.join(directory, 'fscore_dict.pkl'), 'rb') as f:
                fscore_dict = pickle.load(f)
                fscore_dist_all = list(fscore_dict.values())
        else:
            if mesh_dict is None:
                print('prepare for fscore')
                mesh_dict, path_dict, smplx_dict = get_meshes_llava(args.folder)
            if summary_dict is None:
                if os.path.exists(os.path.join(directory, 'summary_dict.pkl')):
                    summary_dict = pickle.load(open(os.path.join(directory, 'summary_dict.pkl'), 'rb'))
                else:
                    raise ValueError('Must calculate chamfer first')
            fscore_dist_all, fscore_dict = calculate_and_return_fscore(mesh_dict, smplx_dict, directory, summary_dict)
            
        output['fscore'] = {
            'mean': torch.tensor(fscore_dist_all).mean().item(),
            'std': torch.tensor(fscore_dist_all).std().item(),
            'median': torch.tensor(fscore_dist_all).median().item(),
            'min': torch.tensor(fscore_dist_all).min().item(),
            'max': torch.tensor(fscore_dist_all).max().item(),
        }
        
    # create file if not exists
    if not os.path.exists(os.path.join(directory, 'summary_dict.json')):
        with open(os.path.join(directory, 'summary_dict.json'), 'w') as f:
            json.dump({}, f)

    with open(os.path.join(directory, 'summary_dict.json'), 'r') as f:
        content = json.load(f)
        
    for metric in args.metrics:
        content[metric] = output[metric]
        
    if 'chamfer' in args.metrics:
        content['failed'] = {
            'total': len(output['failed']),
            'list': output['failed'],
        }
    
    with open(os.path.join(directory, 'summary_dict.json'), 'w') as f:
        json.dump(content, f, indent=4, sort_keys=True)