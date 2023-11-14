# OverlapNetVLAD

This repository represents the official implementation of the paper:

**OverlapNetVLAD: A Coarse-to-Fine Framework for LiDAR-based Place Recognition**


OverlapNetVLAD is a coase-to-fine framework for LiARD-based place recognition, which use global descriptors to propose place candidates, and use overlap prediction to determine the final match.

[[Paper]](https://arxiv.org/abs/2303.06881)

## Instructions

This code has been tested on Ubuntu 18.04 (PyTorch 1.12.1, CUDA 10.2, GeForce GTX 1080Ti).

Pretrained models in [here](https://drive.google.com/drive/folders/1LEGhH38SB9Y7ia_ovYtQ3NzqRMfwJCt1?usp=sharing).

### Requirments

We use *spconv-cu102=2.1.25*, other version may report error. 

The rest requirments are comman and easy to handle.

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install spconv-cu102==2.1.25	
pip install pyymal tqdm open3d 
```

## Extract features

```shell
python tools/utils/gen_bev_features.py
```

## Train

The training of backbone network and overlap estimation network please refs to [BEVNet](https://github.com/lilin-hitcrt/BEVNet). Here is the training of global descriptor generation network.

```shell
python train/train_netvlad.py
```

## Evalute

```shell
python evaluate/evaluate.py
```

the function **evaluate_vlad** is the evaluation of the coarse seaching method using global descriptors.

## Acknowledgement

Thanks to the source code of some great works such as [pointnetvlad](https://github.com/mikacuy/pointnetvlad), [PointNetVlad-Pytorch
](https://github.com/cattaneod/PointNetVlad-Pytorch), [OverlapTransformer](https://github.com/haomo-ai/OverlapTransformer) and so on.


## Citation

If you find this repo is helpful, please cite:


```
@InProceedings{Fu_2023_OverlapNetVLAD,
author = {Fu, Chencan and Li, Lin and Peng, Linpeng and Ma, Yukai and Zhao, Xiangrui and Liu, Yong},
title = {OverlapNetVLAD: A Coarse-to-Fine Framework for LiDAR-based Place Recognition},
journal={arXiv preprint arXiv:2303.06881},
year={2023}
}
```

## Todo

- [x] upload pretrained models
- [ ] add pictures
- [ ] ...