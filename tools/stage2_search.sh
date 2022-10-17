CUDA_VISIBLE_DEVICES=2 python search.py --search_stage second \
 --batch_size 2 --dataset GID --model_name DNAS --checkname search/second_stage --num_worker 8 \
 --alpha_epoch 20 --layers 14 --model_encode_path ../model/model_encode/second_connect.npy \
 --model_core_path ../model/model_encode/core_path.npy
