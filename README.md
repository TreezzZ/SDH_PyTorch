# Supervised Discrete Hashing

## REQUIREMENTS
`pip install -r requirements.txt`

1. pytorch >= 1.0
2. loguru

## DATASETS
1. [cifar10-gist.mat](https://pan.baidu.com/s/1qE9KiAOTNs5ORn_WoDDwUg) password: umb6
2. [cifar-10_alexnet.t](https://pan.baidu.com/s/1ciJIYGCfS3m0marQvatNjQ) password: f1b7
3. [nus-wide-tc21_alexnet.t](https://pan.baidu.com/s/1YglFwoxB-3j7xTEyAc8ykw) password: vfeu
4. [imagenet-tc100_alexnet.t](https://pan.baidu.com/s/1ayv4wdtCOzEDsJy01SjRew) password: 6w5i

## USAGE
```
usage: run.py [-h] [--dataset DATASET] [--root ROOT]
              [--code-length CODE_LENGTH] [--max-iter MAX_ITER]
              [--num-anchor NUM_ANCHOR] [--num-train NUM_TRAIN]
              [--num-query NUM_QUERY] [--topk TOPK] [--gpu GPU] [--seed SEED]
              [--evaluate-interval EVALUATE_INTERVAL] [--lamda LAMDA]
              [--nu NU] [--sigma SIGMA]

SDH_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name.
  --root ROOT           Path of dataset
  --code-length CODE_LENGTH
                        Binary hash code length.(default:
                        12,16,24,32,48,64,128)
  --max-iter MAX_ITER   Number of iterations.(default: 5)
  --num-anchor NUM_ANCHOR
                        Number of anchor.(default: 1000)
  --topk TOPK           Calculate map of top k.(default: all)
  --gpu GPU             Using gpu.(default: False)
  --seed SEED           Random seed.(default: 3367)
  --evaluate-interval EVALUATE_INTERVAL
                        Evaluation interval.(default: 1)
  --lamda LAMDA         Hyper-parameter.(default: 1)
  --nu NU               Hyper-parameter.(default: 1e-5)
  --sigma SIGMA         Hyper-parameter. 2e-3 for cifar-10-gist, 5e-4 for
                        others.
```

## EXPERIMENTS

cifar-10-gist: GIST features, 1000 query images, 5000 training images, sigma=2e-3, map@ALL.

cifar-10-alexnet. Alexnet features, 1000 query images, 5000 training images, sigma=5e-4, map@ALL.

nus-wide-tc21-alexnet. Alexnet features, top 21 classes, 2100 query images, 10500 training images, sigma=5e-4, map@5000.

imagenet-tc100-alexnet: Alexnet features, top 100 classes, 5000 query images, 10000 training images, sigma=5e-4, map@1000.

 bits | 12 | 16 | 24 | 32 | 48 | 64 | 128
   :-:   |  :-:    |   :-:   |   :-:   |   :-:   |   :-:   |   :-:   |   :-:     
cifar-10-gist@ALL | 0.3964 | 0.4335 | 0.4357 | 0.4611 | 0.4729 | 0.4826 | 0.4973
cifar-10-alexnet@ALL | 0.4966 | 0.4837 | 0.5209 | 0.5373 | 0.5411 | 0.5629 | 0.5750
nus-wide-tc21-alexnet@5000 | 0.7504 | 0.7684 | 0.7745 | 0.7932 | 0.7912 | 0.8035 | 0.8162
imagenet-tc100-alexnet@1000 | 0.3529 | 0.4166 | 0.4790 | 0.5096 | 0.5429 | 0.5586 | 0.5974
