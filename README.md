# DNAS
Code for paper：

DNAS: Decoupling Neural Architecture Search for High-Resolution Remote Sensing Image Semantic Segmentation.

Methods' framework
![framework](./paper/framework.jpg)
## How to search
### First Stage
* Create first stage connections: **create_model_encode.py**

  **input:** layers number - 1, **output:** 'model_encode/first{}.npy'


* Train the first surperNet: **train_search.py/train-search-flexinet.sh**
  
  **input:** model_name=flexinet, search_stage=first, model_encode_path=first{}.npy

---
### Second Stage
* Decode the first result and create second stage connections:**first_decoder.py/create_model_encode.py**

  **input:** betas_path, **output:** 'model_encode/second{}.npy'


* Train the second surperNet: **train_search.py/train-search-flexinet.sh**

  **input:** model_name=flexinet, search_stage=second, model_encode_path=second{}.npy
---
### Third Stage
* Decode the second result and create third stage connections:**second_decoder.py/create_model_encode.py**

  **input:** betas_path_stage1, betas_path_stage2, **output:** 'model_encode/third{}.npy'


* Train the third surperNet: **train_search.py/train-search-flexinet.sh**

  **input:** model_name=flexinet, search_stage=third, model_encode_path=third{}.npy

## How to Retrain
* Decode the third result and create retrain cell structure: **third_decoder.py**

  **input:** alphas_path, **output:** 'cell_operations.npy'
* Train the last model: **retrain_nas.py/retrain.sh**

  **input:** model_name=flexinet

## How to Predict


Predict result samples
![framework](./paper/result.jpg)