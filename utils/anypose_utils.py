import copy
import os
import math
import random
import torch
import torch.nn as nn
from torch.nn import Embedding
# import torch.functional as F
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from pytorch3d.ops import knn_gather, knn_points
from utils.cloth_and_material import get_single_face_edges

def get_rest_garemnt_info(smpl_pkl_dict_rest, smpl_layer, rest_garment, is_smplx=False):
    if not is_smplx:
        smpl_output = smpl_layer(
            global_orient=smpl_pkl_dict_rest['global_orient'],
            body_pose=smpl_pkl_dict_rest['body_pose'],
            betas=smpl_pkl_dict_rest['betas'],
            transl=smpl_pkl_dict_rest['transl'],
        )
    else:
        smpl_output = smpl_layer.forward_simple(
            betas=smpl_pkl_dict_rest['betas'],
            full_pose=smpl_pkl_dict_rest['poses'],
            transl=smpl_pkl_dict_rest['transl']
        )

    rest_A = smpl_output.A
    rest_smpl_V = smpl_output.vertices

    smpl_based_lbs_weight = diffuse_lbs_to_space2(
        rest_garment, rest_smpl_V, # s
        smpl_layer.lbs_weights.unsqueeze(0).clone()
    )

    return rest_A, rest_smpl_V, smpl_based_lbs_weight



def calculate_smpl_based_transform_warp(
        smpl_pkl_dict, smpl_layer, rest_smpl_A, rest_garment, smpl_based_lbs_weight, is_smplx=False
):  
    if not is_smplx:
        smpl_output = smpl_layer(
            global_orient=smpl_pkl_dict['global_orient'],
            body_pose=smpl_pkl_dict['body_pose'],
            betas=smpl_pkl_dict['betas'],
            transl=smpl_pkl_dict['transl'],
        )
    else:
        smpl_output = smpl_layer.forward_simple(
            betas=smpl_pkl_dict['betas'],
            full_pose=smpl_pkl_dict['poses'],
            transl=smpl_pkl_dict['transl']
        )

    extra_inp_dict = edict(
        smpl_A=smpl_output.A,
        smpl_A_rest=rest_smpl_A,
        vertices_rest=rest_garment,
        smpl_based_lbs_weight=smpl_based_lbs_weight
    )

    smpl_based_vertices = get_smpl_based_transform(extra_inp_dict)
    smpl_based_vertices = smpl_based_vertices + smpl_pkl_dict['transl'].unqueeze(1)
    
    return smpl_based_vertices




def get_rest_garemnt_info_easy(smpl_pkl_dict_rest, smpl_layer, rest_garment):
    device = smpl_layer.lbs_weights.device
    # print(list(smpl_pkl_dict_rest.keys()))
    # rest_A = torch.from_numpy(smpl_pkl_dict_rest.smplx_A).float().to(device)
    rest_A = None
    if 'smplx_vertices' in smpl_pkl_dict_rest:
        rest_smpl_V = torch.from_numpy(smpl_pkl_dict_rest['smplx_vertices']).float().to(device)
    else:
        rest_smpl_V = torch.from_numpy(smpl_pkl_dict_rest['vertices']).unsqueeze(0).float().to(device)

    smpl_based_lbs_weight = diffuse_lbs_to_space2(
        rest_garment, rest_smpl_V, # s
        smpl_layer.lbs_weights.unsqueeze(0).clone()
    )

    return rest_A, rest_smpl_V, smpl_based_lbs_weight



def calculate_smpl_based_transform_warp_easy(
        smpl_pkl_dict, rest_garment, smpl_based_lbs_weight
):  
    smpl_A = torch.from_numpy(smpl_pkl_dict['A_smplx']).float().to(smpl_based_lbs_weight.device)

    # print('smpl_A', smpl_A[0, :3])
    # assert False
    extra_inp_dict = edict(
        smpl_A=smpl_A.unsqueeze(0),
        smpl_A_rest=None,
        vertices_rest=rest_garment,
        smpl_based_lbs_weight=smpl_based_lbs_weight
    )

    smpl_based_vertices = get_smpl_based_transform(extra_inp_dict)
    return smpl_based_vertices






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


def get_smpl_based_transform(extra_inp_dict):
    smpl_A_raw, smpl_A_rest, vertices_rest = extra_inp_dict['smpl_A'], \
        extra_inp_dict['smpl_A_rest'], extra_inp_dict['vertices_rest']
    
    batch_size, seq_len, K, _, _ = smpl_A_raw.shape
    smpl_A0 = smpl_A_raw[:, :, 0]
    
    if smpl_A_rest is not None:
        smpl_A_rest_inv = transformation_inv(smpl_A_rest.reshape(-1, 4, 4)).reshape(batch_size, K, 4, 4)
        smpl_A = torch.einsum('blkij,bkjq->blkiq', smpl_A_raw, smpl_A_rest_inv)
    else:
        smpl_A = smpl_A_raw

    vertices_new = []
    for bid in range(batch_size):
        smpl_based_lbs_weight = extra_inp_dict['smpl_based_lbs_weight'][bid] # num_v x 21
        smpl_A_bid = smpl_A[bid] # seq_len x 55 x 4 x 4
        smpl_A_raw_bid = smpl_A_raw[bid]
        if True:
            smpl_based_lbs_weight = smpl_based_lbs_weight[:, :22]
            smpl_A_bid = smpl_A_bid[:, :22]
            smpl_A_raw_bid = smpl_A_raw_bid[:, :22]
            smpl_based_lbs_weight = smpl_based_lbs_weight / (smpl_based_lbs_weight.sum(dim=1, keepdim=True) + 1e-8)

            # print('vertices_rest', vertices_rest[bid].shape, smpl_based_lbs_weight.shape, smpl_A_bid.shape)
            
            vertices_new_i = linear_blending(
                vertices_rest[bid], R=None, t=None, lbs_weight=smpl_based_lbs_weight, 
                transform_matrix=smpl_A_bid
            )
            vertices_new_i = torch.cat(
                [vertices_new_i, torch.ones_like(vertices_new_i[..., -1:])], dim=-1
            )[:, :, :3]
            # vertices_new_i = torch.einsum('bij,blj->bli', smpl_A0[bid], vertices_new_i)[:, :, :3]
            vertices_new.append(vertices_new_i.detach())

    return vertices_new

        
def linear_blending(vertices_rest, R, t, lbs_weight, transform_matrix=None):
    # vertices_rest: N x 3
    
    if transform_matrix is None:
        batch_size = R.shape[0]
        K = R.shape[1]
        
        transform_matrix = torch.zeros(batch_size, K, 4, 4, device=vertices_rest.device)
        transform_matrix[:, :, :3, :3] = R
        transform_matrix[:, :, :3, 3] = t
        transform_matrix[:, :, 3, 3] = 1

    # print('vertices_homo', vertices_rest.shape, transform_matrix.shape, lbs_weight.shape)
    if len(vertices_rest.shape) == 2:
        vertices_homo = torch.cat([vertices_rest, torch.ones(vertices_rest.shape[0], 1, device=vertices_rest.device)], dim=-1)
        vertices_homo = torch.einsum('bkij,nj->bnki', transform_matrix, vertices_homo)
    else:
        batch, N, _ = vertices_rest.shape
        vertices_homo = torch.cat([vertices_rest, torch.ones(batch, N, 1, device=vertices_rest.device)], dim=-1)
        vertices_homo = torch.einsum('bkij,bnj->bnki', transform_matrix, vertices_homo)

    vertices_homo = (vertices_homo * lbs_weight.unsqueeze(-1)).sum(dim=2)
    
    vertices_new = vertices_homo[:, :, :3] / vertices_homo[:, :, [3]]
    assert torch.abs(vertices_homo[:, :, [3]] - 1).max() < 1e-3, \
        (torch.abs(vertices_homo[:, :, [3]] - 1).max(), torch.abs(lbs_weight.sum(dim=-1) - 1).max())
    
    return vertices_new


def linear_blending_batch(self, vertices_rest, lbs_weight, transform_matrix, is_normal=False):
    # vertices_rest: b x N x 3, transform_matrix: b x s x N x 4 x 4, lbs_weight: b x N x K
    # print('vertices_rest, lbs_weight, transform_matrix', vertices_rest.shape, lbs_weight.shape, transform_matrix.shape)
    batch_size, N, _ = vertices_rest.shape
    seq_len = transform_matrix.shape[1]
    if not is_normal:
        vertices_homo = torch.cat([vertices_rest, torch.ones(batch_size, N, 1, device=vertices_rest.device)], dim=-1)
    else:
        vertices_homo = torch.cat([vertices_rest, torch.zeros(batch_size, N, 1, device=vertices_rest.device)], dim=-1)
        
    vertices_homo = torch.einsum('blkij,bnj->blnki', transform_matrix, vertices_homo)

    vertices_homo = (vertices_homo * lbs_weight.unsqueeze(1).unsqueeze(-1)).sum(dim=-2)
    vertices_homo = vertices_homo.reshape(batch_size*seq_len, N, 4)
    
    if not is_normal:
        vertices_new = vertices_homo[:, :, :3] / vertices_homo[:, :, [3]]
        assert torch.abs(vertices_homo[:, :, [3]] - 1).max() < 1e-5, \
            (torch.abs(vertices_homo[:, :, [3]] - 1).max(), torch.abs(lbs_weight.sum(dim=-1) - 1).max())
    else:
        vertices_new = vertices_homo[:, :, :3]
    
    return vertices_new.reshape(batch_size, seq_len, N, 3)



def diffuse_lbs_to_space2(points, vertices_rest, lbs_weight, K=16, no_foot=False, return_dist=False):
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


def calculate_pinned_v(vertice, smpl_joints, smpl_verts, lower_garment=True):
    # return
    if lower_garment:
        smpl_root_z = smpl_joints[0, 2]
        smpl_root_x = smpl_joints[0, 0]
        pinned_vs = []
        
        angle = torch.atan2(vertice[:, 2] - smpl_root_z, vertice[:, 0] - smpl_root_x)
        for i in range(6):
            min_angle = 2 * np.pi / 6 * i - np.pi
            max_angle = 2 * np.pi / 6 * (i + 1) - np.pi

            mask = (angle > min_angle) & (angle <= max_angle)

            vertice_tmp = vertice.clone()
            vertice_tmp[~mask] = -10
            pinned_idx = torch.argmax(vertice_tmp[:, 1])
            pinned_vs.append(pinned_idx.item())

    else:
        left_shoulder = smpl_joints[16] + torch.tensor([0, 0.03, 0], device=smpl_joints.device)
        right_shoulder = smpl_joints[17] + torch.tensor([0, 0.03, 0], device=smpl_joints.device)

        closest_idx_left = torch.argmin(torch.norm(vertice - left_shoulder, dim=-1))
        closest_idx_right = torch.argmin(torch.norm(vertice - right_shoulder, dim=-1))

        pinned_vs = [closest_idx_left.item(), closest_idx_right.item()]
    
    closest_idx_on_smpl = []
    for pinned_v in pinned_vs:
        dist_v_smpl = torch.norm(vertice[pinned_v] - smpl_verts, dim=-1)
        closest_idx_on_smpl.append(
            torch.argmin(dist_v_smpl).item()
        )
    
    return pinned_vs, closest_idx_on_smpl



def calculate_pinned_v_dense(vertices, faces, smpl_joints, smpl_verts, lower_garment=True):
    smpl_root_z = smpl_joints[0, 2]
    smpl_root_x = smpl_joints[0, 0]
    pinned_vs = []
    
    vertices_backup = vertices.clone()

    angle = torch.atan2(vertices[:, 2] - smpl_root_z, vertices[:, 0] - smpl_root_x)
    for i in range(2):
        min_angle = np.pi * i - np.pi
        max_angle = np.pi * (i + 1) - np.pi

        mask = (angle > min_angle) & (angle <= max_angle)

        vertices = vertices_backup.clone()
        vertices[~mask] = -10
        if lower_garment:
            max_height = (vertices[:, 1]).max()

            all_indices = torch.arange(vertices.shape[0], device=vertices.device)
            edge_vertices_mask = vertices[:, 1] > max_height - 0.03
            pinned_vs += all_indices[edge_vertices_mask].cpu().numpy().tolist()

        else:
            left_shoulder = smpl_joints[16] + torch.tensor([0, 0.05, 0], device=smpl_joints.device)
            right_shoulder = smpl_joints[17] + torch.tensor([0, 0.05, 0], device=smpl_joints.device)

            closest_idx_left = torch.argmin(torch.norm(vertices - left_shoulder, dim=-1))
            closest_idx_right = torch.argmin(torch.norm(vertices - right_shoulder, dim=-1))

            pinned_vs_now = [closest_idx_left.item(), closest_idx_right.item()]

            if vertices[pinned_vs_now[0], 1] < left_shoulder[1] - 0.06 and vertices[pinned_vs_now[1], 1] < right_shoulder[1] - 0.06:
                pinned_vs_now = []
                max_height = (vertices[:, 1]).max()
                all_indices = torch.arange(vertices.shape[0], device=vertices.device)
                edge_vertices_mask = vertices[:, 1] > max_height - 0.05
                pinned_vs_now += all_indices[edge_vertices_mask].cpu().numpy().tolist()
            
            pinned_vs += pinned_vs_now
    
    vertices = vertices_backup
    closest_idx_on_smpl = []
    for pinned_v in pinned_vs:
        dist_v_smpl = torch.norm(vertices[pinned_v] - smpl_verts, dim=-1)
        closest_idx_on_smpl.append(
            torch.argmin(dist_v_smpl).item()
        )
    
    return pinned_vs, closest_idx_on_smpl



class GeometryModifierCutHands:
    """
    Cut the hands from the SMPL-X mesh. This is done by removing the vertices of the hands and changing the faces
    accordingly. The hands get often stuck in the clothing during simulation and are the cause of many simulation
    failures.
    """

    def __init__(self, hand_removal_data_fname='./data/hand_removal.npz'):
        self.hand_removal_data_fname = hand_removal_data_fname
        self.hand_vids_to_remove, self.wrist_right_vids, self.wrist_left_vids, self.updated_faces = \
            self.load_hand_removal_data()
        self.vertices_to_keep = np.setdiff1d(np.arange(10475), self.hand_vids_to_remove)

    def load_hand_removal_data(self):
        data = np.load(self.hand_removal_data_fname)
        return [data[k] for k in ['hand_vids_to_remove', 'wrist_right_vids', 'wrist_left_vids', 'faces_after_removal']]

    def cut_hands(self, smplx_vertices):
        v = smplx_vertices[self.vertices_to_keep, :]
        v = np.vstack((v, v[self.wrist_left_vids, :].mean(axis=0), v[self.wrist_right_vids, :].mean(axis=0)))
        return v

    def cut_hands_batch(self, smplx_vertices):
        v = smplx_vertices[:, self.vertices_to_keep, :]
        v = torch.cat(
            (v, 
             v[:, self.wrist_left_vids, :].mean(dim=1, keepdim=True), 
             v[:, self.wrist_right_vids, :].mean(dim=1, keepdim=True)), dim=1
        )
        return v

    def get_smplx_new_lbs(self, lbs_old):
        print('lbs_old', lbs_old.shape)
        lbs_new = lbs_old[self.vertices_to_keep]
        # return lbs_new
        lbs_new = torch.cat(
            (lbs_new,
             lbs_old[self.wrist_left_vids].mean(dim=0, keepdim=True), 
             lbs_old[self.wrist_right_vids].mean(dim=0, keepdim=True)), dim=0
        )

        return lbs_new

