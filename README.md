# PointNL

Cascaded Non-local Neural Network for Point Cloud Semantic Segmentation [paper](https://arxiv.org/abs/2007.15488)

### Requirements:

- environment:

  ```
  Ubuntu 18.04
  ```

- install python package: 

  ```
  ./Anaconda3-5.1.0-Linux-x86_64.sh
  ```
  
- install PyTorch :

  ```
  conda install pytorch==1.2.0
  ```




### Usage:

- build the ops

  ```
  cd lib/pointops && python setup.py install && cd ../../
  ```

- Train

  Download the S3IDS dataset and generate superpoints.
  
  We will update the detailed scripts of data processing later. The generation of superpoints can also refer to :link:https://github.com/loicland/superpoint_graph (SPGraph, CVPR 18).
  
  ```
  sh tool/train.sh DATASET_PATH
  ```


- Test

  ```
  sh tool/test.sh DATASET_PATH
  ```
### Citation:
```
  @article{cheng2020cascaded,
  title={Cascaded Non-local Neural Network for Point Cloud Semantic Segmentation},
  author={Cheng, Mingmei and Hui, Le and Xie, Jin and Yang, Jian and Kong, Hui},
  journal={arXiv preprint arXiv:2007.15488},
  year={2020}
}
```

