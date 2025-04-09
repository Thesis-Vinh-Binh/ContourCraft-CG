python evaluation_scripts/dress_4d_evaluate_pose.py --method llava --path runs/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_noseg_openai_Dress4D_eva_crop --is_fscore 1

python evaluation_scripts/dress_4d_evaluate_pose.py --method sewformer --path /is/cluster/fast/sbian/github/sewformer/Sewformer/outputs/Dress4D/Detr2d-V6-final-dif-ce-focal-schd-agp --is_fscore 1
python evaluation_scripts/dress_4d_evaluate_pose.py --method dresscode --path /is/cluster/fast/sbian/github/DressCode/outputs/241106-17-09-43 --is_fscore 1
python evaluation_scripts/dress_4d_evaluate_pose.py --method gpt4o --path /is/cluster/fast/scratch/sbian/Dress4D_eva/gpt4o_preds_sewing --is_fscore 1

python evaluation_scripts/dress_4d_evaluate_pose.py --method garmentrecovery --path /is/cluster/fast/sbian/github/GarmentRecovery/fitting-data-dress4d/garment