# PointNL

Cascaded Non-local Neural Network for Point Cloud Semantic Segmentation:link:https://arxiv.org/abs/2007.15488

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


