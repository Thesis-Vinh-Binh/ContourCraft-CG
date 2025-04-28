# ContourCraft for ChatGarment

This repo is a refined version of the original [ContourCraft Repo](https://github.com/Dolorousrtur/ContourCraft). This code is used to generate simulation results in [ChatGarment](https://chatgarment.github.io/).
It has the following advantages:

1. Allows simulation from arbitrary body mesh and starting pose.
2. Adds blender rendering support.
3. Includes codes to evaluate ChatGarment results on [Dress4D](https://eth-ait.github.io/4d-dress/) and [CloSE](https://github.com/anticdimi/CloSe) datasets.

## Installation and Preparations
Follow [ContourCraft Repo](https://github.com/Dolorousrtur/ContourCraft) to install the environment and download data required for running ContourCraft.


## Inference
Here we give example codes to run inference using the example data provided in this [link](https://drive.google.com/file/d/1QXezA3J6uXqWHGATmcw3jaYxRXY2Ctte/view?usp=sharing). Unpack them to the ``assets`` folder.
### 1. Run simulation
```bash
python simulation_example.py # run simulation and save results
```

### 2. Mesh extraction
```bash
python get_smplx_mesh.py # get smplx mesh
python get_mesh_from_simulation.py # get garment mesh
```

### 3. Blender rendering
```bash
# Rendering a single frame from extracted meshes
/is/cluster/fast/sbian/data/blender-3.6.14-linux-x64/blender --background --python blender_rendering.py -- --obj_dir exp/example_simulation/motion_0 --frameid 60
```

```bash
# Rendering all frames from extracted meshes
python blender_rendering_wrap.py
```

## Extra
### 1. Files in ``generate_new_garments`` folder:
These scripts are used to generate ChatGarment Dataset.

### 2. Files in ``evaluation_scripts`` folder:
These scripts are used to evaluate the outputs of ChatGarment and other methods.

