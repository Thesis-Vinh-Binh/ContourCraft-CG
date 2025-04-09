import pickle as pkl
import os
import torch
import numpy as np
import sys
from tqdm import tqdm

from pytorch3d.structures import Meshes
from pytorch3d.io import IO
from pytorch3d.structures.meshes import join_meshes_as_scene
import argparse
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, required=True, help="path to the save resules shapenet dataset")
parser.add_argument("--garment_folder_path", type=str, default='', help="path to the save resules shapenet dataset")
parser.add_argument("--seed", type=int, default=-1, help="path to the save resules shapenet dataset")
parser.add_argument("--single", action='store_true', default=False, help="path to the save resules shapenet dataset")
parser.add_argument("--video", action='store_true', default=False, help="path to the save resules shapenet dataset")
args = parser.parse_args()


def get_all_paths_garmentcode():
    motion_dir = "/is/cluster/fast/sbian/data/bedlam_motion_for_blender/"
    saved_folder = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v4'

    saved_folder_final = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v4'
    sampled_clothes_list_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/sampled_clothes_v4_combine_all.pkl'
    with open(sampled_clothes_list_path, 'rb') as f:
        sampled_clothes_list = pkl.load(f)

    return motion_dir, saved_folder, saved_folder_final, sampled_clothes_list

if args.index >= 0:
    motion_dir, saved_folder, saved_folder_final, sampled_clothes_list = get_all_paths_garmentcode()
    sampled_clothes_dict = sampled_clothes_list[args.index]
    
    if args.seed > 0:
        saved_folder_final_i = os.path.join(saved_folder, f'{args.index}_{args.seed}', 'motion_0')
    else:
        saved_folder_final_i = os.path.join(saved_folder, f'{args.index}', 'motion_0')

    mesh_saved_path = os.path.join(saved_folder_final_i, 'meshes')
    if not os.path.exists(mesh_saved_path):
        os.makedirs(mesh_saved_path)

    print('mesh_saved_path', mesh_saved_path)

    all_npz_files = [item for item in os.listdir(saved_folder_final_i) if item.endswith('.npz')]
    assert len(all_npz_files) == 1, f"{saved_folder_final_i}, all_npz_files: {all_npz_files}"
    npz_file = all_npz_files[0]

    if 'upper_garment' in sampled_clothes_dict and (not args.single):
        upper_garment = sampled_clothes_dict['upper_garment']
        print('upper_garment', upper_garment)
        upper_name = upper_garment.split('/')[-1]

        if os.path.exists(os.path.join(upper_garment, f'{upper_name}_sim_connected.obj')):
            upper_garmen_obj = os.path.join(upper_garment, f'{upper_name}_sim_connected.obj')
        else:
            upper_garmen_obj = os.path.join(upper_garment, f'{upper_name}_sim.obj')

        obj_path = os.path.join(saved_folder_final_i, 'combined_garment.obj')
        # mesh_upper = trimesh.load(upper_garmen_obj, process=True)
        mesh_upper = IO().load_mesh(upper_garmen_obj)
        mesh_upper = trimesh.Trimesh(vertices=mesh_upper.verts_packed().numpy(), faces=mesh_upper.faces_packed().numpy(), process=True)

        mesh_upper_vertices = mesh_upper.vertices
        mesh_upper_faces = mesh_upper.faces
        len_mesh_upper_vertices = len(mesh_upper_vertices)
        len_mesh_upper_faces = len(mesh_upper_faces)

    else:
        len_mesh_upper_vertices = None
        len_mesh_upper_faces = None
    
    npz_path = os.path.join(saved_folder_final_i, npz_file)
    traj_dict = np.load(npz_path)

else:
    mesh_saved_path = os.path.join(args.garment_folder_path, 'meshes')
    saved_folder_final_i = args.garment_folder_path

    all_npz_files = [item for item in os.listdir(saved_folder_final_i) if item.endswith('.npz')]
    assert len(all_npz_files) == 1, f"{saved_folder_final_i}, all_npz_files: {all_npz_files}"
    npz_file = all_npz_files[0]

    npz_path = os.path.join(saved_folder_final_i, npz_file)
    traj_dict = np.load(npz_path)

    len_mesh_upper_faces = traj_dict['start_face_indices']
    len_mesh_upper_vertices = traj_dict['start_vertex_indices']
    


start_index = 2

pos = traj_dict['pred']
cloth_faces = traj_dict['cloth_faces']

if len(cloth_faces.shape) == 3:
    cloth_faces = cloth_faces[0]

pos = torch.from_numpy(pos).float()
cloth_faces = torch.from_numpy(cloth_faces).long()

seq_len = pos.shape[0]

all_verts = []
if args.video:
    frame_indices = [i + start_index for i in range(300)]
else:
    frame_indices = [i for i in range(start_index, seq_len, 30)]

for i in frame_indices:
    if 'upper_garment' in sampled_clothes_dict and (not args.single):
        upper_verts = pos[i][:len_mesh_upper_vertices]
        upper_faces = cloth_faces[:len_mesh_upper_faces]

        mesh_cloth = Meshes([upper_verts], [upper_faces])

        IO().save_mesh(
            mesh_cloth, 
            os.path.join(mesh_saved_path, f"upper_{i-start_index}.obj")
        )

        lower_verts = pos[i][len_mesh_upper_vertices:]
        lower_faces = cloth_faces[len_mesh_upper_faces:] - len_mesh_upper_vertices

        mesh_cloth = Meshes([lower_verts], [lower_faces])

        IO().save_mesh(
            mesh_cloth, 
            os.path.join(mesh_saved_path, f"lower_{i-start_index}.obj")
        )

    elif 'upper_garment' in sampled_clothes_dict:
        upper_verts = pos[i]
        upper_faces = cloth_faces
        mesh_cloth = Meshes([upper_verts], [upper_faces])

        IO().save_mesh(
            mesh_cloth, 
            os.path.join(mesh_saved_path, f"lower_{i-start_index}.obj")
        )

    else:
        upper_verts = pos[i]
        upper_faces = cloth_faces
        mesh_cloth = Meshes([upper_verts], [upper_faces])

        IO().save_mesh(
            mesh_cloth, 
            os.path.join(mesh_saved_path, f"wholebody_{i-start_index}.obj")
        )


        
