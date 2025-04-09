source ~/.bashrc

python generate_new_garments/try_inference_garmentcode_warp.py --index $1
python generate_new_garments/get_mesh_from_simulation_warp.py --index $1
python generate_new_garments/get_smplx_mesh_warp.py --index $1
python generate_new_garments/blender_rendering_warp.py --index $1