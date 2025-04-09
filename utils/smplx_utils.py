import numpy as np
import pickle as pk

import torch
import torch.nn.functional as F

from pytorch3d.ops import knn_gather, knn_points
import random

def transformation_inv(transforms):
    transforms_shape = transforms.shape
    transforms = transforms.reshape(-1, 4, 4)
    # transforms: batch x 4 x 4
    batch_size = transforms.shape[0]
    rotmat = transforms[:, :3, :3]
    transl = transforms[:, :3, 3]
    rotmat_inv = rotmat.transpose(1, 2)
    
    transl_inv = torch.einsum('bij,bj->bi', rotmat_inv, -transl)
    transforms_inv = torch.cat([rotmat_inv, transl_inv.unsqueeze(-1)], dim=-1)
    transforms_inv = torch.cat(
        [transforms_inv, torch.tensor([[[0, 0, 0, 1]]], device=transforms.device).repeat(batch_size, 1, 1)], dim=1)
    return transforms_inv.reshape(transforms_shape)
    

def do_transformation(V, transforms, R=None, t=None):
    # V: b x N x 3, transforms: b x 4 x 4
    V_shape = V.shape
    V = V.reshape(-1, 3)
    V = torch.cat([V, torch.ones_like(V[:, :1])], dim=-1).reshape(V_shape[0], V_shape[1], 4)
    if transforms is None:
        transforms = torch.eye(4, device=V.device).reshape(1, 4, 4).expand(V_shape[0], 4, 4).clone()
        transforms[:, :3, :3] = R
        transforms[:, :3, 3] = t
        
    V = torch.einsum('bij,bnj->bni', transforms, V)
    return V[:, :, :3]


def do_transformation2(V, transforms, R=None, t=None):
    # V: b x N x 3, transforms: b x L x 4 x 4
    V_shape = V.shape
    V = V.reshape(-1, 3)
    V = torch.cat([V, torch.ones_like(V[:, :1])], dim=-1).reshape(V_shape[0], V_shape[1], 4)
    if transforms is None:
        batch_size, seq_len, _, _ = R.shape
        transforms = torch.eye(4, device=V.device).reshape(1, 1, 4, 4).expand(batch_size, seq_len, 4, 4).clone()
        transforms[:, :, :3, :3] = R
        transforms[:, :, :3, 3] = t
        
    V = torch.einsum('blij,bnj->blni', transforms, V)
    return V[:, :, :, :3]



def get_modified_garment2(vertices0, smpl_verts0, A_smpl0, A_smpl1, smpl_based_lbs, extra_disp):
    # vertices0: n x 3, 
    # A_smpl0, A_smpl1: batch x K x 4 x 4, 
    # smpl_based_lbs: n x K
    batch_size, K, _, _ = A_smpl0.shape
    A_smpl0_inv = transformation_inv(A_smpl0.reshape(-1, 4, 4)).reshape(batch_size, K, 4, 4)

    dists, idx, _ = knn_points(vertices0.unsqueeze(0), smpl_verts0.unsqueeze(0), K=16)
    score = (1 / torch.sqrt(dists + 1e-8))
    score = score / score.sum(dim=2, keepdim=True) # b x num_p x K

    disp_v = knn_gather(extra_disp, idx)
    disp_v = (disp_v * score.unsqueeze(-1)).sum(dim=-2)

    vertices0_homo = torch.cat(
        [vertices0, torch.ones_like(vertices0[:, :1])], dim=-1
    )
    
    transformed_v1 = torch.einsum('bkij,nj->bnki', A_smpl0_inv, vertices0_homo)
    transformed_v1 = (transformed_v1 * smpl_based_lbs.unsqueeze(-1)).sum(dim=2)

    transformed_v1[:, :, :3] = transformed_v1[:, :, :3] + disp_v

    transformed_v2 = torch.einsum('bkij,bnj->bnki', A_smpl1, transformed_v1)
    transformed_v2 = (transformed_v2 * smpl_based_lbs.unsqueeze(-1)).sum(dim=2)

    transformed_v2 = transformed_v2[:, :, :3] / transformed_v2[:, :, [3]]
    return transformed_v2


def linear_blending(vertices_rest, R, t, lbs_weight, transform_matrix=None):
    # vertices_rest: N x 3
    
    if transform_matrix is None:
        batch_size = R.shape[0]
        K = R.shape[1]
        
        transform_matrix = torch.zeros(batch_size, K, 4, 4, device=vertices_rest.device)
        transform_matrix[:, :, :3, :3] = R
        transform_matrix[:, :, :3, 3] = t
        transform_matrix[:, :, 3, 3] = 1

    if len(vertices_rest.shape) == 2:
        vertices_homo = torch.cat([vertices_rest, torch.ones(vertices_rest.shape[0], 1, device=vertices_rest.device)], dim=-1)
        vertices_homo = torch.einsum('bkij,nj->bnki', transform_matrix, vertices_homo)
    else:
        vertices_homo = torch.cat([
            vertices_rest, torch.ones(vertices_rest.shape[0], vertices_rest.shape[1], 1, device=vertices_rest.device)
        ], dim=-1)
        vertices_homo = torch.einsum('bkij,bnj->bnki', transform_matrix, vertices_homo)
        
    # print('vertices_homo', vertices_homo.shape, lbs_weight.shape)
    vertices_homo = (vertices_homo * lbs_weight.unsqueeze(-1)).sum(dim=2)
    vertices_new = vertices_homo[:, :, :3] / vertices_homo[:, :, [3]]
    
    return vertices_new


def linear_blending_batch(vertices_rest, R, t, lbs_weight, transform_matrix=None):
    # vertices_rest: b x N x 3, each batch use on R and T
    batch_size = vertices_rest.shape[0]
    if transform_matrix is None:
        K = R.shape[1]
        
        transform_matrix = torch.zeros(batch_size, K, 4, 4, device=vertices_rest.device)
        transform_matrix[:, :, :3, :3] = R
        transform_matrix[:, :, :3, 3] = t
        transform_matrix[:, :, 3, 3] = 1

    vertices_homo = torch.cat([vertices_rest, torch.ones(batch_size, vertices_rest.shape[1], 1, device=vertices_rest.device)], dim=-1)
    vertices_homo = torch.einsum('bkij,bnj->bnki', transform_matrix, vertices_homo)
    vertices_homo = (vertices_homo * lbs_weight.unsqueeze(-1)).sum(dim=2)
    
    vertices_new = vertices_homo[:, :, :3] / vertices_homo[:, :, [3]]
    
    return vertices_new



def put_V_back_by_A0(V, verts_mean_posed, max_length, A0_inv=None, transl=0):
    scale = max_length / 0.9
    # print('shapes', V.shape, verts_mean_posed.shape, max_length.shape)
    V = V * scale + verts_mean_posed - transl
    
    if A0_inv is not None:
        # print('A0_inv', V.shape, A0_inv.shape)
        V_last = torch.ones_like(V[..., :1])
        V_homo = torch.cat([V, V_last], dim=-1)
        V = torch.einsum('bij,blj->bli', A0_inv, V_homo)
    
    return V[..., :3]


def put_V_back_by_A0_list(V_list, verts_mean_posed, max_length):
    scale = max_length / 0.9
    batch_size = len(V_list)

    V = [V_list[i] * scale[i] + verts_mean_posed[i] for i in range(batch_size)]
    
    return V


def lbs_weight_nofoot(lbs_weight):
    assert len(lbs_weight.shape) == 3
    assert ((lbs_weight.shape[-1] == 55) or (lbs_weight.shape[-1] == 21)) or (lbs_weight.shape[-1] == 22)
    lbs_weight_new  = lbs_weight.clone()
    lbs_weight_new[:, :, 4] += lbs_weight_new[:, :, 10]
    lbs_weight_new[:, :, 5] += lbs_weight_new[:, :, 11]
    lbs_weight_new[:, :, 4] += lbs_weight_new[:, :, 7]
    lbs_weight_new[:, :, 5] += lbs_weight_new[:, :, 8]

    lbs_weight_new[:, :, 7] = 0
    lbs_weight_new[:, :, 8] = 0
    lbs_weight_new[:, :, 10] = 0
    lbs_weight_new[:, :, 11] = 0

    lbs_weight_new[:, :, 13] += (lbs_weight_new[:, :, 12] + lbs_weight_new[:, :, 15]) * 0.5
    lbs_weight_new[:, :, 14] += (lbs_weight_new[:, :, 12] + lbs_weight_new[:, :, 15]) * 0.5
    lbs_weight_new[:, :, 15] = 0
    lbs_weight_new[:, :, 12] = 0
    return lbs_weight_new


def diffuse_lbs_to_space2(points, vertices_rest, lbs_weight, K=16, no_foot=True, return_dist=False):
    # print('shapes', points.shape, vertices_rest.shape, faces.shape, lbs_weight.shape)
    
    dists, idx, _ = knn_points(points, vertices_rest, K=K)
    score = (1 / torch.sqrt(dists + 1e-8))
    score = score / score.sum(dim=2, keepdim=True) # b x num_p x K
    
    if len(lbs_weight.shape) == 2:
        lbs_weight = lbs_weight.unsqueeze(0).expand(len(points), -1, -1)

    if no_foot:
        lbs_weight_new = torch.zeros_like(lbs_weight)
        flags = torch.ones(55, dtype=torch.bool)
        lbs_weight_new[:, :, 4] = (lbs_weight[:, :, 4] + lbs_weight[:, :, 10] + lbs_weight[:, :, 7])
        lbs_weight_new[:, :, 5] = (lbs_weight[:, :, 5] + lbs_weight[:, :, 11] + lbs_weight[:, :, 8])
        lbs_weight_new[:, :, 13] = lbs_weight[:, :, 13] + (lbs_weight[:, :, 12] + lbs_weight[:, :, 15]) * 0.5
        lbs_weight_new[:, :, 14] = lbs_weight[:, :, 14] + (lbs_weight[:, :, 12] + lbs_weight[:, :, 15]) * 0.5

        flags[[4, 5, 7, 8, 10, 11, 12, 13, 14, 15]] = False
        lbs_weight_new[:, :, flags] = lbs_weight[:, :, flags]

        lbs_weight = lbs_weight_new
        
    lbs_weight_gathered = knn_gather(lbs_weight, idx)
    lbs_weight_gathered = (lbs_weight_gathered * score.unsqueeze(-1)).sum(dim=-2)

    if return_dist:
        return lbs_weight_gathered, dists[:, :, 0] # (N, P1)
    
    return lbs_weight_gathered


def get_one_hot_np(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def get_one_hot(targets, nb_classes, device):
    res = torch.eye(nb_classes, device=device)[
        targets.reshape(-1)
    ]
    return res.reshape(list(targets.shape)+[nb_classes])
