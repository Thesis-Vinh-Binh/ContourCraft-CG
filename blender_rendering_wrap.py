import argparse
import time
import subprocess
import os
import pickle as pkl
import numpy as np
import random


def run_python(obj_dir, frameid):

    process = subprocess.Popen(
        ["/is/cluster/fast/sbian/data/blender-3.6.14-linux-x64/blender", 
        "--background", "--python", "blender_rendering.py", 
        "--", "--obj_dir", str(obj_dir), "--frameid", str(frameid)
        ], stdout=subprocess.PIPE
    )

    process.wait()
    return

def main():
    random.seed(2)
    os.environ['PYTHONHASHSEED'] = str(2)
    np.random.seed(2)
    
    HOOD_PROJECT = os.path.dirname(__file__)
    saved_folder_final = os.path.join(HOOD_PROJECT, 'exp/example_simulation')

    obj_dir = os.path.join(saved_folder_final, 'motion_0/meshes')
    
    suffix = '.obj'
    body_paths = [item for item in os.listdir(obj_dir) if item.startswith('SMPLX-mesh-neutral') and item.endswith(suffix)]
    for body_path in body_paths:
        frameid = int(body_path.split('_')[-1].replace(suffix, ''))
        run_python(obj_dir, frameid)



if __name__ == "__main__":
    main()
