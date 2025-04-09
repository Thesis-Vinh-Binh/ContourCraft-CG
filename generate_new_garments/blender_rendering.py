# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse, sys, os, math, re
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json 
import random
import argparse
import sys
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, required=True, help="path to the save resules shapenet dataset")
parser.add_argument("--frameid", type=int, required=True, help="path to the save resules shapenet dataset")
parser.add_argument("--seed", type=int, default=-1, help="path to the save resules shapenet dataset")
argv = sys.argv
argv = argv[argv.index("--") + 1:]
args0 = parser.parse_args(argv)


def get_all_paths_garmentcode():
    motion_dir = "/is/cluster/fast/sbian/data/bedlam_motion_for_blender/"
    saved_folder = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v4'

    saved_folder_final = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v4'
    sampled_clothes_list_path = '/is/cluster/fast/sbian/github/GET3D/exp/aaa_mesh_registrarion/sampled_clothes_v4_combine_all.pkl'
    with open(sampled_clothes_list_path, 'rb') as f:
        sampled_clothes_list = pkl.load(f)

    return motion_dir, saved_folder, saved_folder_final, sampled_clothes_list


motion_dir, saved_folder, saved_folder_final, sampled_clothes_list = get_all_paths_garmentcode()
random.seed(args0.index + args0.frameid * 1345 + 17532)

def get_name_from_index(index, frameid):
    global sampled_clothes_list, saved_folder_final
    garments_dict = sampled_clothes_list[index]

    if args0.seed > 0:
        saved_folder_final_i = os.path.join(saved_folder, f'{index}_{args0.seed}', 'motion_0')
    else:
        saved_folder_final_i = os.path.join(saved_folder, f'{index}', 'motion_0')

    obj_dir = os.path.join(
        saved_folder_final_i, f'meshes/', 
    )
    objs = [item for item in os.listdir(obj_dir) if item.endswith('.obj')]
    # suffix = objs[0].split('_')[-1]
    suffix = '.obj'
    body_path = os.path.join(
        saved_folder_final_i, f'meshes/SMPLX-mesh-neutral_{frameid}{suffix}', 
    )

    if 'whole_garment' in garments_dict:
        garment_path1 = os.path.join(
            saved_folder_final_i, f'meshes/wholebody_{frameid}{suffix}', 
        )
        garment_path2 = None

        garment_txt_path1 = garments_dict['garment_txt']
        garment_txt_path2 = None
    
    else:
        garment_path1 = os.path.join(
            saved_folder_final_i, f'meshes/lower_{frameid}{suffix}',
        )
        
        garment_path2 = os.path.join(
            saved_folder_final_i, f'meshes/upper_{frameid}{suffix}',
        )

        garment_txt_path1 = garments_dict['lower_txt']
        garment_txt_path2 = garments_dict['upper_txt']
    
    body_txt_path = garments_dict['body_txt']

    return body_path, garment_path1, garment_path2, body_txt_path, garment_txt_path1, garment_txt_path2



body_path, garment_path1, garment_path2, body_txt_path, garment_txt_path1, garment_txt_path2 = \
    get_name_from_index(args0.index, args0.frameid)

if args0.seed > 0:
    output_folder = os.path.join(saved_folder_final, f'{args0.index}_{args0.seed}', f'motion_0/imgs/{args0.frameid}')
else:
    output_folder = os.path.join(saved_folder_final, f'{args0.index}', f'motion_0/imgs/{args0.frameid}')

class argsClass:
    views = 4
    obj_garment = ""
    obj_body = ""
    body_texture_path = body_txt_path
    # output_folder = f"/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode/{args0.index}/motion_0/imgs/{args0.frameid}"
    output_folder = output_folder
    scale = 0.8
    format = 'PNG'
    resolution = 512
    engine = 'CYCLES'

args = argsClass()

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = args.engine
render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
render.image_settings.file_format = args.format  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100
bpy.context.scene.cycles.filter_width = 0.01
bpy.context.scene.render.film_transparent = True

bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.diffuse_bounces = 1
bpy.context.scene.cycles.glossy_bounces = 1
bpy.context.scene.cycles.transparent_max_bounces = 3
bpy.context.scene.cycles.transmission_bounces = 3
bpy.context.scene.cycles.samples = 32
bpy.context.scene.cycles.use_denoising = True


def enable_cuda_devices():
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()

    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
        try:
            cprefs.compute_device_type = compute_device_type
            print("Compute device selected: {0}".format(compute_device_type))
            break
        except TypeError:
            pass

    # Any CUDA/OPENCL devices?
    acceleratedTypes = ['CUDA', 'OPENCL']
    accelerated = any(device.type in acceleratedTypes for device in cprefs.devices)
    print('Accelerated render = {0}'.format(accelerated))

    # If we have CUDA/OPENCL devices, enable only them, otherwise enable
    # all devices (assumed to be CPU)
    print(cprefs.devices)
    for device in cprefs.devices:
        device.use = not accelerated or device.type in acceleratedTypes
        print('Device enabled ({type}) = {enabled}'.format(type=device.type, enabled=device.use))

    return accelerated


enable_cuda_devices()
context.active_object.select_set(True)
bpy.ops.object.delete()

# Import textured mesh
bpy.ops.object.select_all(action='DESELECT')


def bounds(obj, local=False):
    local_coords = obj.bound_box[:]
    om = obj.matrix_world

    if not local:
        worldify = lambda p: om @ Vector(p[:])
        coords = [worldify(p).to_tuple() for p in local_coords]
    else:
        coords = [p[:] for p in local_coords]

    rotated = zip(*coords[::-1])

    push_axis = []
    for (axis, _list) in zip('xyz', rotated):
        info = lambda: None
        info.max = max(_list)
        info.min = min(_list)
        info.distance = info.max - info.min
        push_axis.append(info)

    import collections

    originals = dict(zip(['x', 'y', 'z'], push_axis))

    o_details = collections.namedtuple('object_details', 'x y z')
    return o_details(**originals)

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT



def get_texture(mesh_obj, image_texture_path, get_uv=True, idx=0):
    bpy.ops.object.select_all(action='DESELECT')
    # mesh_obj
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')

    if get_uv:
        # Calculate UV mapping
        bpy.ops.uv.smart_project()

    # Exit Edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create a new image texture
    assert isinstance(image_texture_path, list)
    image_texture_path = image_texture_path[random.randint(0, len(image_texture_path)-1)]

    print('os.path.exists(image_texture_path)', os.path.exists(image_texture_path))
    if not os.path.exists(image_texture_path):
        image_texture_path_list = image_texture_path.split('/')
        path1 = image_texture_path_list[:-1] + ['male', 'skin'] + [image_texture_path_list[-1]]
        path1 = '/' + os.path.join(*path1)
        # print('path1', path1)
        if os.path.exists(path1):
            image_texture_path = path1
        
        path2 = image_texture_path_list[:-1] + ['female', 'skin'] + [image_texture_path_list[-1]]
        path2 = '/' + os.path.join(*path2)
        if os.path.exists(path2):
            image_texture_path = path2
        
    image_texture = bpy.data.images.load(image_texture_path)

    # Create a material
    material = bpy.data.materials.new(name=f"ImageTextureMaterial")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    tex_image = material.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = image_texture

    # Connect the image texture to the material
    material.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    # Assign the material to the mesh
    if mesh_obj.data.materials:
        mesh_obj.data.materials[0] = material
    else:
        mesh_obj.data.materials.append(material)



bpy.ops.import_scene.obj(filepath=garment_path1, use_edges=False, use_smooth_groups=False, split_mode='OFF')
imported_garment1 = bpy.context.selected_objects[0]

bpy.ops.object.select_all(action='DESELECT')
imported_garment1.select_set(True)
bpy.context.view_layer.objects.active = imported_garment1
bpy.ops.object.modifier_add(type='SOLIDIFY')
bpy.context.object.modifiers["Solidify"].offset = 1
bpy.context.object.modifiers["Solidify"].thickness = 0.005
bpy.ops.object.select_all(action='DESELECT')
get_texture(imported_garment1, garment_txt_path1, get_uv=True)

if garment_path2 is not None:
    bpy.ops.import_scene.obj(filepath=garment_path2, use_edges=False, use_smooth_groups=False, split_mode='OFF')
    imported_garment2 = bpy.context.selected_objects[0]

    bpy.ops.object.select_all(action='DESELECT')
    imported_garment2.select_set(True)
    bpy.context.view_layer.objects.active = imported_garment2
    bpy.ops.object.modifier_add(type='SOLIDIFY')
    bpy.context.object.modifiers["Solidify"].offset = 1
    bpy.context.object.modifiers["Solidify"].thickness = 0.005
    bpy.ops.object.select_all(action='DESELECT')
    get_texture(imported_garment2, garment_txt_path2, get_uv=True)

imported_body = bpy.ops.import_scene.obj(filepath=body_path, use_edges=False, use_smooth_groups=False, split_mode='OFF')
imported_body = bpy.context.selected_objects[0]
get_texture(imported_body, args.body_texture_path, get_uv=False)
bpy.ops.object.select_all(action='DESELECT')
imported_body.select_set(True)
bpy.context.view_layer.objects.active = imported_body
bpy.context.object.rotation_euler[0] = np.pi
bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)


for this_obj in bpy.data.objects:
    if this_obj.type == "MESH":
        this_obj.select_set(True)
        bpy.context.view_layer.objects.active = this_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.split_normals()

bpy.ops.object.mode_set(mode='OBJECT')
print(len(bpy.context.selected_objects))

bpy.ops.object.select_all(action='DESELECT')
imported_body.select_set(True)
bpy.context.view_layer.objects.active = imported_body

scale = args.scale
factor = max(imported_body.dimensions[0], imported_body.dimensions[1], imported_body.dimensions[2]) / scale

imported_body.scale[0] /= factor
imported_body.scale[1] /= factor
imported_body.scale[2] /= factor
bpy.ops.object.transform_apply(scale=True)


object_details = bounds(imported_body)
obj_mean = [0.5 * (object_details.x.min + object_details.x.max),
            0.5 * (object_details.y.min + object_details.y.max),
            0.5 * (object_details.z.min + object_details.z.max)]

imported_body.location = (-obj_mean[0], -obj_mean[1], -obj_mean[2])
bpy.ops.object.transform_apply(location=True)


bpy.ops.object.select_all(action='DESELECT')
imported_garment1.select_set(True)
bpy.context.view_layer.objects.active = imported_garment1
imported_garment1.scale[0] /= factor
imported_garment1.scale[1] /= factor
imported_garment1.scale[2] /= factor
bpy.ops.object.transform_apply(scale=True)

imported_garment1.location = (-obj_mean[0], -obj_mean[1], -obj_mean[2])
bpy.ops.object.transform_apply(location=True)

if garment_path2 is not None:
    bpy.ops.object.select_all(action='DESELECT')
    imported_garment2.select_set(True)
    bpy.context.view_layer.objects.active = imported_garment2
    imported_garment2.scale[0] /= factor
    imported_garment2.scale[1] /= factor
    imported_garment2.scale[2] /= factor
    bpy.ops.object.transform_apply(scale=True)

    imported_garment2.location = (-obj_mean[0], -obj_mean[1], -obj_mean[2])
    bpy.ops.object.transform_apply(location=True)


bpy.ops.object.light_add(type='AREA')
light2 = bpy.data.lights['Area']

light2.energy = 30000
bpy.data.objects['Area'].location[2] = 0.5
bpy.data.objects['Area'].scale[0] = 100
bpy.data.objects['Area'].scale[1] = 100
bpy.data.objects['Area'].scale[2] = 100

# Place camera
cam = scene.objects['Camera']
cam.location = (0, 1.2, 0)  # radius equals to 1
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty

scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
cam_constraint.target = cam_empty

stepsize = 360.0 / args.views
rotation_mode = 'XYZ'

img_follder = os.path.join(os.path.abspath(args.output_folder), 'img')
camera_follder = os.path.join(os.path.abspath(args.output_folder), 'camera')
os.makedirs(img_follder, exist_ok=True)
os.makedirs(camera_follder, exist_ok=True)

rotation_angle_list = np.random.rand(args.views)
elevation_angle_list = np.random.rand(args.views)
rotation_angle_list = rotation_angle_list * 360
elevation_angle_list = elevation_angle_list * 30
np.save(os.path.join(camera_follder, 'rotation'), rotation_angle_list)
np.save(os.path.join(camera_follder, 'elevation'), elevation_angle_list)

# creation of the transform.json
to_export = {
    'camera_angle_x': bpy.data.cameras[0].angle_x,
    "aabb": [[-scale/2,-scale/2,-scale/2],
            [scale/2,scale/2,scale/2]]
}
frames = [] 

for i in range(0, args.views):
    get_texture(imported_garment1, garment_txt_path1, get_uv=False)
    if garment_path2 is not None:
        get_texture(imported_garment2, garment_txt_path2, get_uv=False)
    get_texture(imported_body, body_txt_path, get_uv=False)

    cam_empty.rotation_euler[2] = math.radians(rotation_angle_list[i])
    cam_empty.rotation_euler[0] = math.radians(elevation_angle_list[i])

    # print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))
    render_file_path = os.path.join(img_follder, '%03d.png' % (i))
    if os.path.exists(render_file_path):
        continue
    scene.render.filepath = render_file_path
    bpy.ops.render.render(write_still=True)
    # might not need it, but just in case cam is not updated correctly
    bpy.context.view_layer.update()
    print('render_file_path', render_file_path)

    rt = get_3x4_RT_matrix_from_blender(cam)
    pos, rt, scale = cam.matrix_world.decompose()

    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0,0,0,1])
    print(matrix)

    to_add = {\
        "file_path":f'{str(i).zfill(3)}.png',
        "transform_matrix":matrix
    }
    frames.append(to_add)

to_export['frames'] = frames
with open(f'{img_follder}/transforms.json', 'w') as f:
    json.dump(to_export, f,indent=4)    

