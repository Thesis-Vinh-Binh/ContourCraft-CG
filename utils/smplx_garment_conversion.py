# convert the garment to the target pose & shape import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes, Pointclouds

from utils.robust_lbs_weight_utils import get_best_skinning_weights
from utils.smplx_utils import get_modified_garment2

def deform_garments(smplx_layer, params_old, params_new, garment_mesh, lbs_weights, scale=True, return_lbs=False):

    garment_verts = garment_mesh.verts_padded()
    if (garment_verts.max(dim=1)[0] - garment_verts.min(dim=1)[0]).max() > 10:
        scale = True
        garment_verts = garment_verts * 0.01
    
    # assert 'scale'  in params_old
    if 'scale' not in params_old:
        garment_verts = garment_verts - params_old['transl'].unsqueeze(1)
        smplx_out_old = smplx_layer.forward_simple(
            betas=params_old['betas'],
            full_pose=params_old['poses'],
            transl=params_old['transl'] * 0.0,
            pose2rot=True
        )
    
    else:
        smplx_out_old = smplx_layer.forward_simple(
            betas=params_old['betas'],
            full_pose=params_old['poses'],
            transl=torch.zeros(1, 3, device=params_old['betas'].device),
            pose2rot=True,
            scale=params_old['scale']
        )

        body_verts_0 = params_old['body_vs']
        body_verts_1 = smplx_out_old.vertices

        # print(body_verts_0.shape, body_verts_1.shape)
        transl = (body_verts_0.max(dim=1)[0] + body_verts_0.min(dim=1)[0] - body_verts_1.max(dim=1)[0] - body_verts_1.min(dim=1)[0]) * 0.5
        garment_verts = garment_verts - transl.unsqueeze(1)
        params_old['transl'] = transl

        # body_verts_0 = body_verts_0 - body_verts_0.mean(dim=1).unsqueeze(1)
        # body_verts_1 = body_verts_1 - body_verts_1.mean(dim=1).unsqueeze(1)
        # print(body_verts_0.std(dim=1), body_verts_1.std(dim=1))

    A_old = smplx_out_old['A']
    smplx_verts = smplx_out_old.vertices

    # print('A_old', A_old.shape, A_old[:, 0])

    garment_mesh = Meshes(verts=garment_verts, faces=garment_mesh.faces_padded())
    smplx_mesh = Meshes(verts=smplx_verts, faces=smplx_layer.faces_tensor.unsqueeze(0))

    final_lbs_weight, smplx_lbs_weight = get_best_skinning_weights(
        garment_mesh, smplx_mesh, lbs_weights, max_distance=0.05, max_angle=25.0
    )
    final_lbs_weight = torch.from_numpy(final_lbs_weight).float().cuda()
    if torch.isnan(final_lbs_weight).any() or torch.isinf(final_lbs_weight).any():
        final_lbs_weight = torch.from_numpy(smplx_lbs_weight).float().cuda()

    smplx_out_new = smplx_layer.forward_simple(
        betas=params_new['betas'],
        full_pose=params_new['poses'],
        transl=params_new['transl'] * 0.0,
        pose2rot=True
    )
    A_new = smplx_out_new['A']
        
    extra_disp = smplx_layer.get_disp(
        params_new['betas']-params_old['betas'],
        params_old['poses'], params_new['poses'],
    )

    # print('garment_verts', garment_verts.shape)
    # print('smplx_verts', smplx_verts.shape)
    # print('A_new', A_new.shape, A_new[:, 0])
    # print('lbs_weights', lbs_weights.shape)
    # print('extra_disp', extra_disp[0].shape)
    
    vertices_new_bid = get_modified_garment2(
                garment_verts[0], smplx_verts[0], A_old, A_new, final_lbs_weight, extra_disp=extra_disp[0])[0]
    
    vertices_new_bid = vertices_new_bid + params_new['transl']
    if return_lbs:
        return vertices_new_bid, final_lbs_weight
    
    return vertices_new_bid