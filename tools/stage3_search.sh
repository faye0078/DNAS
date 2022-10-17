CUDA_VISIBLE_DEVICES=3 python search.py --search_stage third \
 --batch_size 2 --dataset GID --model_name DNAS --checkname search/third_stage --num_worker 8 \
 --alpha_epoch 20 --layers 14 --model_encode_path ../model/model_encode/third_connect.npy
