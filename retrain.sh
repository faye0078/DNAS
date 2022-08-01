CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 24 --dataset GID --checkname 'retrain/GID15_fastnas' --resize 512 --crop_size 512 --num_worker 8\
 --epochs 200 --model_name 'fast-nas' --nclass 3
