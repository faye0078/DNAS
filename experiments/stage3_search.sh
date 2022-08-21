CUDA_VISIBLE_DEVICES=0 python train_search.py \
 --batch_size 2 --dataset GID --model_name DNAS --checkname search/RSNet --num_worker 8\
 --alpha_epoch 20 
