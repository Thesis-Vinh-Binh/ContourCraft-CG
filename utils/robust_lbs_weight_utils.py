import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from pytorch3d.ops import sample_points_from_meshes, knn_points, knn_gather
from pytorch3d.structures import Meshes
from pytorch3d.ops.laplacian_matrices import cot_laplacian

from utils.smplx_utils import diffuse_lbs_to_space2

def compute_face_normals(meshes):
    verts_packed = meshes.verts_padded()
    faces_packed = meshes.faces_packed()
    verts_packed = verts_packed[faces_packed]
    face_normals = torch.cross(
        verts_packed[:, 1] - verts_packed[:, 0],
        verts_packed[:, 2] - verts_packed[:, 0],
        dim=1,
    )
    face_normals = face_normals / torch.norm(face_normals, dim=1, keepdim=True)
    return face_normals

        
def get_high_conf_indices(mesh_garment, mesh_smpl, max_distance=0.05, max_angle=25.0):
    garment_normals = compute_vertex_normals(mesh_garment)
    smpl_normals = compute_vertex_normals(mesh_smpl)

    verts_garment = mesh_garment.verts_packed()
    verts_smpl = mesh_smpl.verts_packed()
    
    dists, indices, closest_pos = knn_points(verts_garment.unsqueeze(0), verts_smpl.unsqueeze(0), K=1, return_nn=True)
    closest_normal = knn_gather(smpl_normals.unsqueeze(0), indices) # 1 x N x 1 x 3
    
    threshold_distance = calculate_threshold_distance(verts_garment, max_distance)
    garment_data = {
        "position": verts_garment,
        "normal": garment_normals.reshape(-1, 3)
    }
    closest_points_data = {
        "position": closest_pos.reshape(-1, 3),
        "normal": closest_normal.reshape(-1, 3),
    }
    high_confidence_flag = filter_high_confidence_matches_torch(
        garment_data, closest_points_data, threshold_distance, max_angle
    )
    
    return high_confidence_flag


def get_high_conf_indices_knn(mesh_garment, mesh_smpl, max_distance=0.05, max_angle=25.0):
    garment_normals = compute_vertex_normals(mesh_garment)
    smpl_normals = compute_vertex_normals(mesh_smpl)

    verts_garment = mesh_garment.verts_packed()
    verts_smpl = mesh_smpl.verts_packed()
    
    dists, indices, closest_pos = knn_points(verts_garment.unsqueeze(0), verts_smpl.unsqueeze(0), K=1, return_nn=True)
    closest_normal = knn_gather(smpl_normals.unsqueeze(0), indices) # 1 x N x 1 x 3
    
    threshold_distance = calculate_threshold_distance(verts_garment, max_distance)
    garment_data = {
        "position": verts_garment,
        "normal": garment_normals.reshape(-1, 3)
    }
    closest_points_data = {
        "position": closest_pos.reshape(-1, 3),
        "normal": closest_normal.reshape(-1, 3),
    }
    high_confidence_flag = filter_high_confidence_matches_torch(
        garment_data, closest_points_data, threshold_distance, max_angle
    )
    
    return high_confidence_flag, (dists, indices, closest_pos)



def get_best_skinning_weights(mesh_garment, mesh_smpl, lbs_weights, max_distance=0.05, max_angle=25.0):
    # single-mesh operation, batch = 1
    garment_normals = compute_vertex_normals(mesh_garment)
    smpl_normals = compute_vertex_normals(mesh_smpl)
    
    verts_garment = mesh_garment.verts_packed()
    verts_smpl = mesh_smpl.verts_packed()
    num_verts = verts_garment.shape[0]
        
    dists, indices, closest_pos = knn_points(verts_garment.unsqueeze(0), verts_smpl.unsqueeze(0), K=1, return_nn=True)
    
    # indices: 1 x N x 1
    closest_normal = knn_gather(smpl_normals.unsqueeze(0), indices) # 1 x N x 1 x 3
    
    smpl_based_lbs = diffuse_lbs_to_space2(
        verts_garment.unsqueeze(0), verts_smpl.unsqueeze(0), lbs_weights.unsqueeze(0)
    )
    
    threshold_distance = calculate_threshold_distance(verts_garment, max_distance)
    threshold_distance = threshold_distance.cpu().numpy()
    
    L, inv_area = cot_laplacian(verts_garment, mesh_garment.faces_packed())
    L_sum = torch.sparse.sum(L, dim=1).to_dense().cpu().numpy()

    L = L.coalesce().cpu()
    L_indices = L.indices().numpy()
    L_values = L.values().numpy()
    L_size = L.size()
    # print('L_indices', L_values.max(), L_values.min(), inv_area.max(), inv_area.min())
    L = sp.coo_matrix((L_values, (L_indices[0], L_indices[1])), shape=L_size).tocsr()
    L = L - sp.diags(L_sum, offsets=0)
    inv_area = inv_area.reshape(-1).cpu().numpy()
    
    ################################################################################################
    garment_data = {
        "position": verts_garment.cpu().numpy(),
        "normal": garment_normals.reshape(-1, 3).cpu().numpy(),
    }
    
    closest_points_data = {
        "position": closest_pos.reshape(-1, 3).cpu().numpy(),
        "normal": closest_normal.reshape(-1, 3).cpu().numpy(),
    }
    high_confidence_indices = filter_high_confidence_matches(
        garment_data, closest_points_data, threshold_distance, max_angle
    )
    
    # print('high_confidence_indices', len(high_confidence_indices), num_verts)
    
    low_confidence_indices = list(
        set(range(num_verts)) - set(high_confidence_indices)
    )
    
    smpl_based_lbs = torch.clamp(smpl_based_lbs, 0.0, 1.0)
    smpl_based_lbs = smpl_based_lbs / (smpl_based_lbs.sum(dim=-1, keepdim=True) + 1e-6)
    
    smpl_based_lbs_np = smpl_based_lbs[0].cpu().numpy() # N x lbs_weight

    final_lbs_weight = np.zeros((num_verts, smpl_based_lbs_np.shape[1]))
    final_lbs_weight[high_confidence_indices] = smpl_based_lbs_np[high_confidence_indices]

    try:
        final_lbs_weight = do_inpainting(
            high_confidence_indices, low_confidence_indices, final_lbs_weight, L, inv_area)
    except:
        final_lbs_weight = smpl_based_lbs_np
    
    return final_lbs_weight, smpl_based_lbs_np


def compute_vertex_normals(meshes):
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()
    verts_normals = torch.zeros_like(verts_packed)
    vertices_faces = verts_packed[faces_packed]

    faces_normals = torch.cross(
        vertices_faces[:, 2] - vertices_faces[:, 1],
        vertices_faces[:, 0] - vertices_faces[:, 1],
        dim=1,
    )

    verts_normals.index_add_(0, faces_packed[:, 0], faces_normals)
    verts_normals.index_add_(0, faces_packed[:, 1], faces_normals)
    verts_normals.index_add_(0, faces_packed[:, 2], faces_normals)
    
    return torch.nn.functional.normalize(
        verts_normals, eps=1e-6, dim=1
    )


def calculate_threshold_distance(verts_garment, threadhold_ratio=0.05):
    """Returns dbox * 0.05

    dbox is the target mesh bounding box diagonal length.
    """
    
    length = verts_garment.max(dim=0)[0] - verts_garment.min(dim=0)[0]
    # print('length', length)
    length = torch.norm(length)

    threshold_distance = length * threadhold_ratio

    return threshold_distance


def filter_high_confidence_matches(target_vertex_data, closest_points_data, max_distance, max_angle):
    """filter high confidence matches using structured arrays."""

    target_positions = target_vertex_data["position"]
    target_normals = target_vertex_data["normal"]
    source_positions = closest_points_data["position"]
    source_normals = closest_points_data["normal"]

    # Calculate distances (vectorized)
    distances = np.linalg.norm(source_positions - target_positions, axis=1)

    # Calculate angles between normals (vectorized)
    cos_angles = np.einsum("ij,ij->i", source_normals, target_normals)
    cos_angles /= np.linalg.norm(source_normals, axis=1) * np.linalg.norm(target_normals, axis=1)
    cos_angles = np.abs(cos_angles)  # Consider opposite normals by taking absolute value
    angles = np.arccos(np.clip(cos_angles, -1, 1)) * 180 / np.pi

    # Apply thresholds (vectorized)
    high_confidence_indices = np.where((distances <= max_distance) & (angles <= max_angle))[0]

    return high_confidence_indices.tolist()



def filter_high_confidence_matches_torch(target_vertex_data, closest_points_data, max_distance, max_angle):
    """filter high confidence matches using structured arrays."""

    target_positions = target_vertex_data["position"]
    target_normals = target_vertex_data["normal"]
    source_positions = closest_points_data["position"]
    source_normals = closest_points_data["normal"]

    # Calculate distances (vectorized)
    distances = torch.norm(source_positions - target_positions, dim=-1)

    # Calculate angles between normals (vectorized)
    cos_angles = torch.einsum("ij,ij->i", source_normals, target_normals)
    cos_angles = cos_angles / (torch.norm(source_normals, dim=-1) * torch.norm(target_normals, dim=-1))
    cos_angles = torch.abs(cos_angles)  # Consider opposite normals by taking absolute value
    angles = torch.arccos(torch.clamp(cos_angles, min=-1, max=1)) * 180 / np.pi

    # Apply thresholds (vectorized)
    high_confidence_flag = ((distances <= max_distance) & (angles <= max_angle))

    return high_confidence_flag


def do_inpainting(known_indices, unknown_indices, all_weights, L, inv_area):
    num_bones = all_weights.shape[1] ######################## sparse ?
    W = all_weights
    
    Q = -L + L @ sp.diags(inv_area) @ L
    
    # print(Q)

    S_match = np.array(known_indices)
    S_nomatch = np.array(unknown_indices)

    Q_UU = sp.csr_matrix(Q[np.ix_(S_nomatch, S_nomatch)])
    Q_UI = sp.csr_matrix(Q[np.ix_(S_nomatch, S_match)])

    W_I = W[S_match, :] # match_num x num_bones
    W_U = W[S_nomatch, :] # nomatch_num x num_bones
    W_I = np.clip(W_I, 0.0, 1.0) 
    # print('W_I', W_I.min(), W_I.max())

    for bone_idx in range(num_bones):
        if W_I[:, bone_idx].max() < 1e-3:
            continue
        
        b = -Q_UI @ W_I[:, bone_idx]
        W_U[:, bone_idx] = splinalg.spsolve(Q_UU, b)
        
    # print('W_U', W_U.max(), W_U.min())
    W[S_nomatch, :] = W_U

    # apply constraints,
    # each element is between 0 and 1
    W = np.clip(W, 0.0, 1.0)

    # normalize each row to sum to 1
    W_sum = W.sum(axis=1, keepdims=True)
    
    W = W / W.sum(axis=1, keepdims=True)

    return W