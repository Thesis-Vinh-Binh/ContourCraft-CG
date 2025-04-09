import numpy as np

import torch
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix, axis_angle_to_quaternion, quaternion_to_axis_angle

close_type_dict = {
 0: 'Hat',
 1: 'Body',
 2: 'Shirt',
 3: 'TShirt',
 4: 'Vest',
 5: 'Coat',
 6: 'Dress',
 7: 'Skirt',
 8: 'Pants',
 9: 'ShortPants',
 10: 'Shoes',
 11: 'Hoodies',
 12: 'Hair',
 13: 'Swimwear',
 14: 'Underwear',
 15: 'Scarf',
 16: 'Jumpsuits',
 17: 'Jacket'
}

close_type_dict_inverse = {v: k for k, v in close_type_dict.items()}

upper_garments = [
    'Shirt', 'TShirt', 'Vest', 'Coat', 'Hoodies', 'Jacket'
]

lower_garments = [
    'Pants', 'ShortPants', 'Skirt'
]

wholebody_garments = [
    'Dress', 'Jumpsuits'
]

upper_garments_indices = [
    close_type_dict_inverse[item] for item in upper_garments
]

lower_garments_indices = [
    close_type_dict_inverse[item] for item in lower_garments
]
wholebody_garments_indices = [
    close_type_dict_inverse[item] for item in wholebody_garments
]

def get_seged_points(data):
    labels = data['labels'] # V
    points = data['points'] # V x 3
    scale = data['scale']

    print('labels', np.unique(labels))

    # points = points / scale

    upper_garment_flag = np.zeros(labels.shape, dtype=bool)
    lower_garment_flag = np.zeros(labels.shape, dtype=bool)
    wholebody_garment_flag = np.zeros(labels.shape, dtype=bool)

    for i in upper_garments_indices:
        upper_garment_flag[labels == i] = True

    for i in lower_garments_indices:
        lower_garment_flag[labels == i] = True
    
    for i in wholebody_garments_indices:
        wholebody_garment_flag[labels == i] = True
    
    upper_points = points[upper_garment_flag]
    lower_points = points[lower_garment_flag]
    wholebody_points = points[wholebody_garment_flag]

    garment_points = points[upper_garment_flag | lower_garment_flag | wholebody_garment_flag]

    return upper_points, lower_points, wholebody_points, garment_points
    


# pose # (72,) SMPL pose parameters;
# betas # (10,) SMPL shape parameters;
# trans # (3,) SMPL translation parameters;

def motion_generation(data, pose_rest_new, transl_register, restbetas):
    pose = data['pose']
    betas = data['betas']
    trans = data['trans'] * 0.0 + transl_register
    scale = data['scale']

    pose_new = pose_rest_new.copy()
    pose_new[:21] = pose.reshape(24, 3)[:21]
    pose = pose_new

    #### interpolate
    pose_rest_new_quaternion = axis_angle_to_quaternion(torch.from_numpy(pose_rest_new).float().reshape(-1, 3))
    poses_new_quaternion = axis_angle_to_quaternion(torch.from_numpy(pose).float().reshape(-1, 3))

    pose_interp = []
    transl_interp = []
    for i in range(60):
        alpha = i / 61
        pose_interp.append(pose_rest_new_quaternion * (1 - alpha) + poses_new_quaternion * alpha)
        transl_interp.append(transl_register * (1 - alpha) + trans.reshape(3) * alpha)

    pose_interp = torch.stack(pose_interp, dim=0) # N x 55 x 4
    pose_interp = quaternion_to_axis_angle(pose_interp.reshape(-1, 4)).reshape(-1, 55, 3)
    transl_interp = np.stack(transl_interp, axis=0)
    pose_interp[:, 22] = 0 # jaw pose always be zero

    # the first several frames are static
    pose = pose.reshape(1, 55, 3)
    trans = trans.reshape(1, 3)
    poses_new = np.concatenate([pose_interp[:1]] * 4 + [pose_interp] +[pose] * 30, axis=0)
    transl_new = np.concatenate([transl_interp[:1]] * 4 + [transl_interp] + [trans] * 30, axis=0)

    # return poses_new, transl_new, betas
    returned_dict = {
        'betas': restbetas.reshape(1, -1),
        'poses': poses_new,
        'trans': transl_new,
        'scale': scale,
        'betas_smpl': betas,
    }
    return returned_dict
    

