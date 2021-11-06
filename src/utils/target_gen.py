import torch
import mmcv


class Solo_targetGEN:
    def __init__(self,
                 sigma=0.1,
                 strides=(4, 8, 16, 32, 64),
                 grid_num=[40,36,24,16,12],
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
    ):
        self.sigma = sigma
        self.scale_ranges = scale_ranges
        self.strides = strides
        self.grid_num = grid_num


    def centre_of_mass(self, bitmasks):
        """
            마스크 무게중심 계산
        """
        _, h, w = bitmasks.size()
        ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
        xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

        m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
        m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
        m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
        centre_x = m10 / m00
        centre_y = m01 / m00
        return centre_x, centre_y


    def gen_target(self,
                    gt_bboxes_raw,
                    gt_labels_raw,
                    gt_masks_raw,
                    featmap_sizes=None):

        device = gt_labels_raw[0].device

        # area
        gt_areas = torch.sqrt(
            (gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1])
        )

        mask_label_ls = []
        cate_label_ls = []
        mask_ind_label_ls = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid in zip(self.scale_ranges, self.strides, featmap_sizes, self.grid_num):
            mask_label = torch.zeros([num_grid ** 2, featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device)
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            mask_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            # 해당 사이즈 범위의 영역이 없는 경우
            # 모두 비어 있는 라벨을 넣고 continue
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                mask_label_ls.append(mask_label)
                cate_label_ls.append(cate_label)
                mask_ind_label_ls.append(mask_ind_label)
                continue

            # 해당되는 영역이 있다면 진행
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # 마스크 중심점 계산
            gt_masks_pt = gt_masks.to(device=device)
            centre_ws, centre_hs = self.centre_of_mass(gt_masks_pt)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

            output_stride = stride
            for seg_mask, gt_label, half_h, half_w, centre_h, centre_w, valid_mask_flag in \
                                                zip(gt_masks, gt_labels, half_hs, half_ws, centre_hs, centre_ws, valid_mask_flags):

                # valid 마스크가 아니면 생성 x
                if not valid_mask_flag:
                    continue

                # SxS 그리드 상에서의 mask 위치 계산
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)
                coord_w = int((centre_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((centre_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((centre_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((centre_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((centre_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((centre_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                # 해당 그리드 범위에 라벨 값 부여
                cate_label[top:(down+1), left:(right+1)] = gt_label

                # mask 그리기
                seg_mask = mmcv.imrescale(seg_mask.cpu().numpy(), scale=1. / output_stride)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)
                        mask_label[label, :seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        mask_ind_label[label] = True

            mask_label_ls.append(mask_label)
            cate_label_ls.append(cate_label)
            mask_ind_label_ls.append(mask_ind_label)
        return mask_label_ls, cate_label_ls, mask_ind_label_ls


if __name__ == "__main__":
    import sys
    import os.path as osp
    add_p = [osp.dirname(osp.abspath(osp.dirname(__file__)))]
    add_p.append(f'{osp.sep}'.join(add_p[0].split(osp.sep)[:-1]))
    sys.path.extend(add_p)

    import random
    import numpy as np
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2

    from matplotlib import patches
    from matplotlib import pyplot as plt

    from prettyprinter import cpprint
    from src.datasets import COCODataset

    # from augmentation import Augment_HSV, HistogramEqualization, Mosaic


    image_path = '/data/cocodataset/val2017'
    annFile = "/data/cocodataset/annotations/instances_val2017.json"
    dataset = COCODataset(image_path, annFile, None)

    img_size = 512
    transforms = A.Compose([
        # Mosaic(img_size=256, dataset_obj=dataset, p=1),
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1),
        A.Affine(p=1),
        A.Blur(p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    idx = random.randint(1, 100)
    dataset.set_transforms(transforms)
    image, target = dataset.__getitem__(index=idx)

    masks = target['masks']
    print(f'Image Shape : {image.shape}')
    print(f'Mask Shapes : {masks.shape}')
    print()

    for i, (box, label) in enumerate(zip(target['boxes'], target['labels'])):
        cpprint(f'Bbox_{i + 1} : {box}')
        label = dataset.coco_ids_to_class_names[dataset.coco_ids[label.item()]]
        cpprint(f'label_{i + 1} : {label}')
        print()

    for box in target['boxes']:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='red', fill=False)
        ax = plt.gca()
        ax.add_patch(rect)

    for mask in masks:
        mask = np.array(mask)
        plt.imshow(mask)
        plt.show()


    from src.modules.backbone import deformable_resnet50
    from src.modules.neck import FPN
    from src.modules.head import SoloHead

    backbone = deformable_resnet50(pretrained=False)
    backbone.eval()

    fpn = FPN(in_chans=[256, 512, 1024, 2048], mid_chans=256, for_detect=True, use_p6=True)
    fpn.eval()

    head = SoloHead()
    head.eval()

    fake_inp = image.unsqueeze(0)
    backbone_output = backbone(fake_inp)
    fpn_output = fpn(backbone_output)
    mask_output, cate_output = head(fpn_output)

    featmap_sizes = [mask.shape[-2:] for mask in mask_output]
    target_gen = SOLO_TargetGEN()
    mask_label_ls, cate_label_ls, mask_ind_label_ls = \
        target_gen.gen_target(target['boxes'], target['labels'], target['masks'],
                              featmap_sizes=featmap_sizes)
