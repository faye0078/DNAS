CUDA_VISIBLE_DEVICES=3 python retrain.py \
 --batch_size 2 --dataset GID --checkname 'retrain/DNAS' --resize 512 --crop_size 512 --num_worker 8\
 --epochs 200 --model_name 'DNAS' --layers 14 \
 --cell_arch '../model/model_encode/cell_operations.npy' \
 --model_encode_path '../model/model_encode/third_connect.npy' \
