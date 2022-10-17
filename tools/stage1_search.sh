CUDA_VISIBLE_DEVICES=2 python search.py --search_stage first\
 --batch_size 2 --dataset GID --model_name DNAS --checkname search/stage1 --num_worker 8\
 --alpha_epoch 20  --layers 14 --model_encode_path ../model/model_encode/first_connect.npy
