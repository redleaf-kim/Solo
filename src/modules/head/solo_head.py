# https://github.com/WXinlong/SOLO/blob/master/mmdet/models/anchor_heads/solo_head.py


import os
import sys
add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.utils import ConvModule
from src.modules.utils import normal_init, bias_init_with_prob, multi_apply


INF = 1e8


def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


class SoloHead(nn.Module):
    def __init__(self,
                 num_classes=81,
                 in_channels=256,
                 feat_channels=256,
                 stacked_convs=7,
                 cate_down_pos=0,
                 grid_num=[40,36,24,16,12],
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), **kwargs):
        super(SoloHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.cate_down_pos = cate_down_pos

        self.grid_num = grid_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):
        self.cate_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        self.solo_cate = nn.ModuleList([
                nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1) for _ in self.grid_num
        ])
        self.solo_mask = nn.ModuleList([
            nn.Conv2d(self.feat_channels, num**2, 1, padding=0) for num in self.grid_num
        ])


    def init_weights(self):
        for m in self.mask_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        bias_mask = bias_init_with_prob(0.01)
        for m in self.solo_mask:
            normal_init(m, std=0.01, bias=bias_mask)
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)


    def forward(self, feats):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)

        mask_pred, cate_pred = multi_apply(self.forward_single, new_feats,
                                          list(range(len(self.grid_num))),
                                          upsampled_size=upsampled_size)
        return mask_pred, cate_pred


    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5,
                              mode='bilinear', align_corners=True,
                              recompute_scale_factor=True),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:],
                              mode='bilinear', align_corners=True))


    def forward_single(self, x, idx, upsampled_size=None):
        mask_feat = x
        cate_feat = x

        ##############
        # ins branch #
        ##############
        # concat coord to
        x_range = torch.linspace(-1, 1, mask_feat.shape[-1], device=mask_feat.device)
        y_range = torch.linspace(-1, 1, mask_feat.shape[-2], device=mask_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([mask_feat.shape[0], 1, -1, -1])
        x = x.expand([mask_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        mask_feat = torch.cat([mask_feat, coord_feat], 1)

        for i, mask_layer in enumerate(self.mask_convs):
            mask_feat = mask_layer(mask_feat)

        mask_feat = F.interpolate(mask_feat, scale_factor=2, mode='bilinear')
        mask_pred = self.solo_mask[idx](mask_feat)


        ###############
        # cate branch #
        ###############
        for i, cate_layer in enumerate(self.cate_convs):
            if i == self.cate_down_pos:
                seg_num_grid = self.grid_num[idx]
                cate_feat = F.interpolate(cate_feat, size=seg_num_grid, mode='bilinear')
            cate_feat = cate_layer(cate_feat)

        cate_pred = self.solo_cate[idx](cate_feat)
        if not self.training:
            mask_pred = F.interpolate(mask_pred.sigmoid(), size=upsampled_size, mode='bilinear')
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return mask_pred, cate_pred



if __name__ == '__main__':
    from src.modules.backbone import deformable_resnet50
    from src.modules.neck import FPN

    backbone = deformable_resnet50(pretrained=False)
    backbone.eval()

    fpn = FPN(in_chans=[256, 512, 1024, 2048], mid_chans=256, for_detect=True, use_p6=True)
    fpn.eval()

    head = SoloHead()
    head.eval()

    fake_inp = torch.randn(1, 3, 224, 224)
    backbone_output = backbone(fake_inp)

    print('='*15, 'Backbone Output Information', '='*15)
    for output in backbone_output:
        print(f'feature_map_size: {output.shape}')
        print('-'*58)
    print('='*58, '\n\n')

    fpn_output = fpn(backbone_output)
    print('=' * 15, 'FPN Ouput Information', '=' * 15)
    if isinstance(fpn_output, dict):
        for output in fpn_output:
            print(f'feature_map_size - {output} : {fpn_output[output].shape}')
            print('-' * 58)
    elif isinstance(fpn_output, list):
        for idx, output in enumerate(fpn_output):
            print(f'feature_map_size - fpn_output[{idx}] : {output.shape}')
            print('-' * 58)
    else:
        print(f'feature_map_size : {fpn_output.shape}')
    print('='*58, '\n\n')


    mask_output, cate_output = head(fpn_output)
    print('=' * 15, 'Mask Ouput Information', '=' * 15)
    if isinstance(mask_output, dict):
        for output in mask_output:
            print(f'mask_output size - {output} : {mask_output[output].shape}')
            print('-' * 58)
    elif isinstance(mask_output, (list, tuple)):
        for idx, output in enumerate(mask_output):
            print(f'mask_output size - mask_output[{idx}] : {output.shape}')
            print('-' * 58)
    else:
        print(f'mask_output size : {mask_output.shape}')
    print('='*58, '\n\n')


    print('=' * 15, 'Cate Ouput Information', '=' * 15)
    if isinstance(cate_output, dict):
        for output in cate_output:
            print(f'cate_output size - {output} : {cate_output[output].shape}')
            print('-' * 58)
    elif isinstance(cate_output, (list, tuple)):
        for idx, output in enumerate(cate_output):
            print(f'cate_output size - mask_output[{idx}] : {output.shape}')
            print('-' * 58)
    else:
        print(f'cate_output size : {cate_output.shape}')
    print('='*58, '\n\n')