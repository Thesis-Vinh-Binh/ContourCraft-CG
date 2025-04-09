python evaluation_scripts/close_evaluate.py --method llava --path runs/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_noseg_openai_CloSE_eva_crop  --is_fscore 1

python evaluation_scripts/close_evaluate.py --method gpt4o --path /is/cluster/fast/scratch/sbian/CloSE_eva/gpt4o_preds_sewing  --is_fscore 1
python evaluation_scripts/close_evaluate.py --method sewformer --path /is/cluster/fast/sbian/github/sewformer/Sewformer/outputs/close/Detr2d-V6-final-dif-ce-focal-schd-agp  --is_fscore 1
python evaluation_scripts/close_evaluate.py --method dresscode --path /is/cluster/fast/sbian/github/DressCode/outputs/241106-16-56-59  --is_fscore 1