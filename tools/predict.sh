CUDA_VISIBLE_DEVICES=3 python predict.py \
 --batch_size 1 --dataset GID --checkname 'predict' --resize 512 --crop_size 512 --num_worker 8