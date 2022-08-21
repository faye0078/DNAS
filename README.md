# DNAS

Code for paper：

[DNAS: Decoupling Neural Architecture Search for High-Resolution Remote Sensing Image Semantic Segmentation.](https://www.mdpi.com/2072-4292/14/16/3864)

Methods' framework
![framework](./paper/framework.jpg)

---
## How to search
### First Stage
* Create first stage connections: 

```bash
sh ./experiments/stage1_encode.sh
```


* Train the first surpernet: 

```bash
sh ./experiments/stage1_search.sh
```

### Second Stage
* Create second stage connections: 

```bash
sh ./experiments/stage2_encode.sh
```

* Train the second surpernet: 

```bash
sh ./experiments/stage2_search.sh
```
### Third Stage
* Create third stage connections: 

```bash
sh ./experiments/stage3_encode.sh
```
* Train the third surpernet: 

```bash
sh ./experiments/stage3_search.sh
```
---
## How to Retrain
* Decode the third result and create retrain cell structure:
```bash
sh ./experiments/retrain_encode.sh
```
* Train the last model:
```bash
sh ./experiments/retrain.sh
```
---
## How to Predict
```bash
sh ./experiments/predict.sh
```
Predict result samples
![framework](./paper/result.jpg)