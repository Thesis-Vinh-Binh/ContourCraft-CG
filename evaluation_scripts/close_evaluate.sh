conda init
source ~/.bashrc

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate ccraft

cd /workspace/ContourCraft-CG/

# python evaluation_scripts/close_evaluate.py --method llava --folder close_image_scan_img_recon --metrics chamfer fscore
# python evaluation_scripts/close_evaluate.py --method llava --folder close_image_scan_cg_blip --metrics chamfer fscore
python evaluation_scripts/close_evaluate.py --method llava --folder close_image_scan_cg_retrieval --metrics chamfer fscore
# python evaluation_scripts/close_evaluate.py --method d2g --folder d2g_close  --metrics chamfer fscore
# python evaluation_scripts/close_evaluate.py --method d2g --folder d2g_close_caption  --metrics chamfer fscore