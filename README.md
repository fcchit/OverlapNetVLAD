# OverlapNetVLAD

This repository represents the official implementation of the paper:

**OverlapNetVLAD: A Coarse-to-Fine Framework for LiDAR-based Place Recognition**

OverlapNetVLAD is a coase-to-fine framework for LiARD-based place recognition, which use global descriptors to propose place candidates, and use overlap prediction to determine the final match.

## Instructions

This code has been tested on

- Python 3.8.13, PyTorch 1.12.1, CUDA 10.2, GeForce GTX 1080Ti

### Requirments

We use *spconv-cu102=2.1.25*, other version may report error. 

The rest requirments are comman and easy to handle.

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

## Todo

- [ ] upload pretrained models
- [ ] add pictures
- [ ] ...