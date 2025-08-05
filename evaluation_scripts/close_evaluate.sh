# python evaluation_scripts/close_evaluate.py --method llava --folder close_image_scan_img_recon --metrics chamfer fscore
python evaluation_scripts/close_evaluate.py --method llava --folder close_image_scan_blip_img_recon --metrics chamfer fscore

python evaluation_scripts/close_evaluate.py --method d2g --folder d2g_close  --metrics chamfer fscore
python evaluation_scripts/close_evaluate.py --method d2g --folder d2g_close_caption  --metrics chamfer fscore
