import bpy
import random
import numpy as np
import os
import argparse
import sys
import pickle as pkl
import yaml
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, required=True, help="path to the save resules shapenet dataset")
parser.add_argument("--motion_path", type=str, default='', help="path to the save resules shapenet dataset")
parser.add_argument("--garment_folder_path", type=str, default='', help="path to the save resules shapenet dataset")
parser.add_argument("--seed", type=int, default=-1, help="path to the save resules shapenet dataset")
parser.add_argument("--single", action='store_true', default=False, help="path to the save resules shapenet dataset")
parser.add_argument("--is_generator", default=False, action='store_true')
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


def get_name_from_index(index, is_generator):
    motion_dir, saved_folder, saved_folder_final, sampled_clothes_list = get_all_paths_garmentcode()
    
    if args.single:
        skirt_path = 'exp/all_used_indices_skirts_dresses.json'
        with open(skirt_path, 'r') as f:
            all_skirt_indices_dict = json.load(f)
        
        all_skirt_indices = list(all_skirt_indices_dict.keys())
        all_skirt_indices = sorted([int(item) for item in all_skirt_indices])
        print('all_skirt_indices', len(all_skirt_indices))
        sampled_clothes_dict = sampled_clothes_list[all_skirt_indices[index]]
    else:
        sampled_clothes_dict = sampled_clothes_list[index]

    motion_path = sampled_clothes_dict['motion_path']
    motion_path = os.path.join(motion_dir, motion_path)
    
    if not is_generator:
        body_name = sampled_clothes_dict['body_name']
        
        if body_name == 'mean_all_apart':
            motion_path = motion_path.replace('_300.npz', '_300_apart.npz')

    if args.seed > 0:
        saved_path = os.path.join(saved_folder_final, f'{args.index}_{args.seed}', 'motion_0', 'meshes')
    else:
        saved_path = os.path.join(saved_folder_final, f'{args.index}', 'motion_0', 'meshes')

    objs = [item for item in os.listdir(saved_path) if item.startswith('wholebody')]
    wholebody_flag = (len(objs) > 0)
    return saved_path, motion_path, wholebody_flag

if args.index >= 0:
    saved_path, motion_path, wholebody_flag = get_name_from_index(args.index, args.is_generator)
else:
    motion_path = args.motion_path
    saved_path = os.path.join(args.garment_folder_path, 'meshes')

print('motion_path', motion_path)

start_frame = 0


bpy.ops.preferences.addon_enable(module='smplx_blender_addon')

bpy.ops.object.smplx_add_animation(
    filepath=motion_path,
)

endframe = bpy.context.scene.frame_end
print('endframe', endframe)
object_names = [obj.name for obj in bpy.data.objects]
print('object_names', object_names)
mesh_name = 'SMPLX-mesh-neutral'
mesh_obj = bpy.data.objects[mesh_name]


if args.video:
    frame_indices = [i + start_frame for i in range(300)]
else:
    frame_indices = [i for i in range(start_frame, endframe, 30)]

for frame_number in tqdm(frame_indices, dynamic_ncols=True):
    print('frame_number', frame_number)
    bpy.context.scene.frame_set(frame_number+1)

    mesh_obj = bpy.data.objects[mesh_name]
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)

    output_mesh_path = os.path.join(saved_path, f"{mesh_name}_{frame_number-start_frame}.obj")
    bpy.ops.wm.obj_export(filepath=output_mesh_path, export_selected_objects=True)

    print(f"Mesh '{mesh_name}' at frame {frame_number} has been exported to {output_mesh_path}")