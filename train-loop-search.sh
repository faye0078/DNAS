CUDA_VISIBLE_DEVICES=3 python train_loop_search.py \
 --batch-size 2 --dataset uadataset --checkname search --num_worker 4\
 --alpha_epoch 0 --filter_multiplier 8 --resize 512 --crop_size 512\
 --resume /media/dell/DATA/wy/Seg_NAS/run/uadataset/search/experiment_2/1_stage3_epoch5_checkpoint.pth.tar