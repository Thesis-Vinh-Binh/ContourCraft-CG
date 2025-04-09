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


HOOD_PROJECT = os.path.dirname(__file__)
saved_folder = os.path.join(HOOD_PROJECT, 'exp/example_simulation')
motion_path = 'assets/male_31_us_1190_0022_300.npz'

saved_path = os.path.join(saved_folder, 'motion_0/meshes')

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