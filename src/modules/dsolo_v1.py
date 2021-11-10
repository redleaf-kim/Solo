import os
import sys

add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)


import torch
import torch.nn as nn
import torch.nn.functional as F

import modules
from src.eval import get_masks


class DSolo_v1(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(DSolo_v1, self).__init__()

        self.backbone = getattr(modules, cfg['backbone'])(**cfg.get('backbone_args', {}))
        self.neck = modules.FPN(**cfg.get('fpn_args', {}))
        self.head = getattr(modules, cfg['head'])(**cfg.get('head_args', {}))
        self.post_process = getattr(modules, cfg['post_process'])(cfg['post_process_args'],
                                     grid_num=cfg['head_args']['grid_num'],
                                     strides=cfg['head_args']['strides'],
                                     num_classes=cfg['head_args']['num_classes'])
        self.solo_loss = getattr(modules, cfg['loss'])(**cfg.get('loss_args', {}))


    def forward(self, inp, targets=None, img_metas=None):
        if isinstance(inp, tuple):
            inp = torch.stack(inp)

        backbone_out = self.backbone(inp)
        neck_out = self.neck(backbone_out)
        mask_x_pred, mask_y_pred, cate_preds = self.head(neck_out)
        if self.training:
            gt_bboxes_list = [target['boxes'] for target in targets]
            gt_labels_list = [target['labels'] for target in targets]
            gt_masks_list = [target['masks'] for target in targets]
            losses = self.solo_loss(mask_x_pred, mask_y_pred, cate_preds, gt_bboxes_list, gt_labels_list, gt_masks_list)
            return losses
        else:
            return self.post_process(mask_x_pred, mask_y_pred, cate_preds, img_metas=img_metas)



if __name__ == '__main__':
    import yaml
    import random
    import os.path as osp
    import albumentations as A

    from torchinfo import summary
    from torch.utils.data import DataLoader
    from albumentations.pytorch.transforms import ToTensorV2
    from pathlib import Path
    from omegaconf import OmegaConf
    from src.datasets import COCODataset


    def collate_fn(batch):
        return tuple(zip(*batch))

    root = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    root = Path(root)
    conf_dir  = root / 'configs' / 'models'
    conf_name = 'light_dsolo_v1'

    model_cfg_p = conf_dir / f'{conf_name}.yaml'
    model_cfg = OmegaConf.load(root / model_cfg_p)


    image_path = '/data/cocodataset/val2017'
    annFile = "/data/cocodataset/annotations/instances_val2017.json"
    dataset = COCODataset(image_path, annFile, None)

    img_size = 512
    transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1),
        A.Affine(p=1),
        A.Blur(p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    idx = random.randint(1, 100)
    dataset.set_transforms(transforms)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    for batch in dataloader:
        images, targets, img_metas = batch
        break


    print('model config:\n', model_cfg)
    model = DSolo_v1(model_cfg)

    summary(model)

    device = 'cpu'
    mode = "valid"

    model = model.to(device)
    if mode != "train":
        model.eval()
        with torch.no_grad():
            output = model(torch.stack(images), img_metas=img_metas)
    else:
        output = model(torch.stack(images), targets=targets)


    if mode == "train":
        print('=' * 5, 'DSOLO V1 Train Ouput Information', '=' * 4)
        for k, v in output.items():
            print(f'{k}: {v}')
            print('-' * 42)
        print('=' * 42)
    else:
        print('=' * 5, 'DSOLO V1 Test Ouput Information', '=' * 4)
        for out in output:
            mask, cate, cate_score = out

            print("Mask shape: ", mask.shape)
            print("Cate shape: ", cate.shape)
            print("Cate Score shape: ", cate_score.shape)