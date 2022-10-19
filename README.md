# DNAS

Code for paper：

[DNAS: Decoupling Neural Architecture Search for High-Resolution Remote Sensing Image Semantic Segmentation.](https://www.mdpi.com/2072-4292/14/16/3864)

**Abstract**: In DNAS, a hierarchical search space with three levels is recommended: path-level, connection-level, and cell-level. To adapt to this hierarchical search space, we devised a new decoupling search optimization strategy to decrease the memory occupation. More specifically, the search optimization strategy consists of three stages: (1) a light super-net (i.e., the specific search space) in the path-level space is trained to get the optimal path coding; (2) we endowed the optimal path with various cross-layer connections and it is trained to obtain the connection coding; (3) the super-net, which is initialized by path coding and connection coding, is populated with kinds of concrete cell operators and the optimal cell operators are finally determined. It is worth noting that the well-designed search space can cover various network candidates and the optimization process can be done efficiently.

**Methods' framework**

![framework](./paper/framework.jpg)
---

## Requirement

Ubuntu(or other Linux distribution), one GPU (video memory greater than 12GB)

* python>=3.8.13
* numpy>=1.22.3
* pytorch>=1.10.0
* pillow>=9.0.1
* opencv>=4.5.4
* tqdm>=4.62.3
* torchstat>=0.0.7

## Dataset

We use the [GID-5](https://captain-whu.github.io/GID/)(4 bands: R, G, B, and NIR) dataset in this rep. The original image of size 6800 × 7200 and the corresponding label are cut into blocks of size 512 × 512. These blocks are randomly divided into a training set, a validation set, and a test set in a ratio of 6 : 2 : 2. 

The list file in [list_dir](./data/lists/GID/). You can Download these blocks from [OneDrive](https://1drv.ms/u/s!AkdG3kpBQQcHg8BVUajKSwLF3WeNNg?e=gy3xI0) or [BaiduNetDisk](https://pan.baidu.com/s/1fLXmJZiJ7STPX2jh4S9nRg)(code: 1111), and move it to the [data](./data/) dir 

## Model Zoo

|   Methods   |  mIoU  | GFLOPs | Params |  Memory  |                            Model                             |
| :---------: | :----: | :----: | :----: | :------: | :----------------------------------------------------------: |
| DNAS (L=12) | 0.8917 | 16.89  | 6.15 M | 753.0 M  | [OneDrive](https://1drv.ms/u/s!AkdG3kpBQQcHg8BWclILK1DFdiR9Rw?e=TlocZ5) or [BaiduNetDisk](https://pan.baidu.com/s/17izJilQRBydyapN2TobflA)(code: 1111) |
| DNAS (L=14) | 0.9140 | 54.06  | 7.14 M | 1195.4 M | [OneDrive](https://1drv.ms/u/s!AkdG3kpBQQcHg8BX4s0uysjCmoZDIQ?e=EJDzbt) or [BaiduNetDisk](https://pan.baidu.com/s/1iYC5AW0L67HCjgoNLSsiVA)(code: 1111) |

## Simple Use the Searched and Trained Model

Take DNAS (L=14) model as an example, download it and move to the [model_encode](./model/model_encode/) dir

* Test model mIoU, GFLOPS, params, and memory

```bash
cd tools && python test_retrain_model.py
```

* Use the trained model to predict

```bash
sh predict.sh
```

**Predict result samples:**

![framework](./paper/result.jpg)
(a) image  (b) ground truth  (c) PSPNet  (d) Deeplabv3+  (e) HRNet  (f) MSFCN  (g) Auto-deeplab  (h) Fast-NAS  (i) DNAS.
## Train the Searched Model on Target Dataset

Take DNAS (L=14) model and GID-5 dataset as an example, download model encode file and move to the [model_encode](./model/model_encode/) dir. 

* Make a personal dataset in dataloaders dir

```python
class GIDDataset(Dataset)
```

* Run the Retrain

```
sh retrain.sh
```

## Search Model on Target Dataset

* Search process(Need change some config path in .sh file between command line)

```bash
cd tools
sh stage1_encode.sh # Create first stage connections
sh stage1_search.sh # Train the first surpernet
sh stage2_encode.sh # Create second stage connections
sh stage2_search.sh # Train the second surpernet
sh stage3_encode.sh # Create third stage connections
sh stage3_search.sh # Train the third surpernet
sh retrain_encode.sh # Decode the third result and create retrain cell structure
```

* Finally, retrain the searched model:

```bash
sh retrain.sh
```

## Citation

Consider cite the DNAS in your publications if it helps your research. 

```
@article{rs14163864,
    author = {Wang Yu and Li, Yansheng and Chen, Wei and Li, Yunzhou and Dang, Bo},
    title = {DNAS: Decoupling Neural Architecture Search for High-Resolution Remote Sensing Image Semantic Segmentation},
    journal = {Remote Sensing},
    url = {https://www.mdpi.com/2072-4292/14/16/3864},
    doi = {10.3390/rs14163864}
}
```

Consider cite this project in your publications if it helps your research. 

```
@misc{DynamicRouting,
    author = {Wang Yu},
    title = {DNAS},
    howpublished = {\url{https://github.com/faye0078/DNAS}},
    year ={2022}
}
```

## Contact
If you have any questions about it, please let me know. (📧 email: wangfaye@whu.edu.cn)
