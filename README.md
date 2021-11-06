# SOLO: Segmenting Objects by Locations

Implementation of SOLO Networks for Instance Segmentation based on torchvision.


> [**SOLO: Segmenting Objects by Locations**](https://arxiv.org/abs/1912.04488),
> Xinlong Wang, Tao Kong, Chunhua Shen, Yuning Jiang, Lei Li
> In: Proc. European Conference on Computer Vision (ECCV), 2020
> *arXiv preprint ([arXiv 1912.04488](https://arxiv.org/abs/1912.04488))*

<br></br>
## TODO
- [x] Solo Version1 Implement.
- [ ] Training on Coco dataset.
- [ ] Decoupled Solo Version1 Implement.
- [ ] Demo Code implementation


<br></br>
## ⚙ Getting Start <a name = 'GettingStart'></a>
### Directory structure
```
├─data
│    ├─cocodataset
│         ├─train2017
│         ├─val2017
│         ├─test2017
│         └─annotations
└─workspace
    ├─configs
    ├─scripts
    └─src
        ├─datasets
        ├─eval
        ├─models
        ├─modules
        │    ├─backbone
        │    ├─head
        │    ├─neck
        │    ├─utils
        │    └─solo_v1.py
        └─utils
```

<br></br>
### Config
- You can set various experimental environments in `configs/experiments/*.yaml` and  `configs/models/*.yaml`

``` yaml
experiment:
  name: 'Solo_r50_fpn_3x'
  group: "v1"
  log_model: true
  wandb: true
  log_step: true

model: configs/models/solo_v1.yaml

general:
  seed: 822
  gpus: [0]
  epoch: 36
  precision: 32
  data_root: ''
  trn_img_size:
    height: 512
    width: 512
  val_img_size:
    height: 512
    width: 512

  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]

  save_top_k: 5
  save_interval: 1

optim:
  name: 'SGD'
  args:
    lr: 0.01
    momentum: 0.9
    weight_decay: 0.0001


grad_clip:
  use: true
  max_norm: 35
  norm_type: 2


sched:
  name: 'MultiStepLR'
  args:
    milestones: [27, 33]
    gamma: 0.1


train_loader:
  image_path: 'cocodataset/train2017'
  annFile: 'cocodataset/annotations/instances_train2017.json'
  batch_size: 8
  num_workers: 8
  shuffle: trues


valid_loader:
  image_path: 'cocodataset/val2017'
  annFile: 'cocodataset/annotations/instances_val2017.json'
  batch_size: 2
  num_workers: 8
  shuffle: false
```


<br></br>
### Reference
- https://arxiv.org/abs/1912.04488
- https://github.com/WXinlong/SOLO