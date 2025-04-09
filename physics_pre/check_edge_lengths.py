import pytorch3d
from pytorch3d.io import IO
import torch

path1 = '/is/cluster/fast/sbian/github/HOOD/hood_data/aux_data/garment_meshes/longskirt.obj'
path2 = '/ps/scratch/ps_shared/sbian/hood_simulation_garmentcode_v2_physics_hood/88_3/motion_0/combined_garment.obj'

mesh1 = IO().load_mesh(path1)
mesh2 = IO().load_mesh(path2)

edges1 = mesh1.edges_packed()
verts1 = mesh1.verts_packed()

edge1_len = torch.norm(verts1[edges1[0]] - verts1[edges1[1]], dim=-1)

edges2 = mesh2.edges_packed()
verts2 = mesh2.verts_packed()

edge2_len = torch.norm(verts2[edges2[0]] - verts2[edges2[1]], dim=-1)

print('edge_len1', edge1_len.mean(), edge1_len.max(), edge1_len.min())
print('edge_len2', edge2_len.mean(), edge2_len.max(), edge2_len.min())