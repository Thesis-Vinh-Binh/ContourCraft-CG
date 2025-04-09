import os
import json
import numpy as np
from tqdm import tqdm


motion_dir = "/is/cluster/fast/sbian/data/bedlam_motion_for_blender/"
all_motions = sorted([item for item in os.listdir(motion_dir) if item.endswith('_300.npz')])
print('all_motions', len(os.listdir(motion_dir)))

choosed_motion_paths = []

for i, motion_name in enumerate(tqdm(all_motions, dynamic_ncols=True)):
    # print('i', i)
    # print('motion_name', motion_name)
    motion_path = os.path.join(motion_dir, motion_name)
    data = np.load(motion_path)
    poses = data['poses'].reshape(-1, 55, 3)[30:]

    leg_poses = poses[:, [1, 2, 4, 5]]
    # print('leg_poses', leg_poses.shape)
    leg_poses_std = np.std(leg_poses, axis=0)
    # print('leg_poses_std', leg_poses_std.mean())

    if leg_poses_std.mean() > 0.05:
        choosed_motion_paths.append(motion_path)

print('choosed_motion_paths', len(choosed_motion_paths))
with open('exp/choosed_motion_paths_loose.json', 'w') as f:
    json.dump(choosed_motion_paths, f)