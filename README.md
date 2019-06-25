# Supervised Discrete Hashing

论文[Supervised Discrete Hashing](http://openaccess.thecvf.com/content_cvpr_2015/html/Shen_Supervised_Discrete_Hashing_2015_CVPR_paper.html)

## Requirements
1. pytorch 1.1
2. loguru

## 数据集下载
1. [CIFAR10-gist](https://pan.baidu.com/s/1qE9KiAOTNs5ORn_WoDDwUg) 密码：umb6
2. [NUS-WIDE](https://pan.baidu.com/s/1S1ZsYCEfbH5eQguHs8yG_w)
密码：4839

## 运行
`python run.py --dataset cifar10-gist --data-path <data_path> --code-length 64 `

日志记录在`logs`文件夹内


## 参数说明
```
SDH_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset used to train (default: cifar10)
  --data-path DATA_PATH
                        Path of cifar10 dataset
  --code-length CODE_LENGTH
                        Binary hash code length (default: 12)
  --max-iter MAX_ITER   Maximum iteration number (default: 5)
  --num-anchor NUM_ANCHOR
                        Number of anchor points (default: 1000)
  --num-query NUM_QUERY
                        Number of query(default: 1000)
  --num-train NUM_TRAIN
                        Number of train(default: 5000)
  --topk TOPK           Compute map of top k (default: -1, use whole dataset)
  --evaluate-freq EVALUATE_FREQ
                        Frequency of evaluate (default: 1)
  --lamda LAMDA         Hyper-parameter, regularization term weight (default:
                        1.0)
  --nu NU               Hyper-parameter, penalty term of hash function output
                        (default: 1e-5)
  --sigma SIGMA         Hyper-parameter, rbf kernel width (default: 0.4)
  --gpu GPU             Use gpu (default: 0. -1: use cpu)
  --batch-size BATCH_SIZE
                        Batch size (default: 128)
  --num-workers NUM_WORKERS
                        Number of loading data workers (default: 0)

```

# 实验
 bits | 12 | 24 | 32 | 64 | 128 
   :-:   |  :-:    |   :-:   |   :-:   |   :-:   |    :-:  
cifar-10 mAP | 0.3629  | 0.4241  | 0.4260  | 0.4571  | 0.4755 
