CUDA_VISIBLE_DEVICES=0 python train_search.py \
 --batch-size 2 --dataset GID --checkname 'search/RSNet' --num_worker 8\
 --alpha_epoch 20 --filter_multiplier 8 --resize 512 --crop_size 512\
 --model_name RSNet