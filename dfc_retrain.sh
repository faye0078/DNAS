CUDA_VISIBLE_DEVICES=3 python retrain_dfc.py \
 --batch-size 20 --dataset uadataset_dfc --checkname 'retrain/loop_1' --resize 512 --crop_size 512 --num_worker 4\
 --epochs 100 --model_name 'flexinet' --nclass 12\
 --resume /media/dell/DATA/wy/Seg_NAS/run/uadataset_dfc/retrain/loop_1/experiment_1/epoch2_checkpoint.pth.tar