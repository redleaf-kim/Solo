import os
import os.path

import cv2
import torch
import numpy as np

from prettyprinter import cpprint
from typing import Any, Callable, Optional, Tuple, List

from pycocotools.coco import COCO

from torch.utils.data import Dataset


COCO_CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')


class COCODataset(Dataset):
    """`Custom MS Coco Detection Dataset.
    Args:
        image_path (string): Image path where images are downloaded to.
        annFile (string): Path to json annotation file.
        transforms (callable, optional): albumentation transform function
    """

    CLASSES = COCO_CLASSES

    def __init__(
        self,
        image_path: str,
        annFile: str,
        transforms: Optional[Callable] = None
    ) -> None:
        super().__init__()

        self.coco = COCO(annFile)

        whole_image_ids = list(sorted(self.coco.imgs.keys()))
        self.ids = []
        self.no_anno_list = []

        for idx in whole_image_ids:
            annotations_ids = self.coco.getAnnIds(imgIds=idx, iscrowd=False)
            annotations = self.coco.loadAnns(annotations_ids)
            if len(annotations_ids) == 0:
                self.no_anno_list.append(idx)
            if self._has_only_empty_bbox(annotations):
                self.no_anno_list.append(idx)
            else:
                self.ids.append(idx)

        self.coco_ids = sorted(self.coco.getCatIds())
        self.coco_ids_to_continuous_ids = {coco_id: i for i, coco_id in enumerate(self.coco_ids)}
        self.coco_ids_to_class_names = {category['id']: category['name'] for category in self.coco.loadCats(self.coco_ids)}

        self.image_path = image_path
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)
        # return 1200

    def set_transforms(self, transforms):
        self.transforms = transforms

    def _has_only_empty_bbox(self, annotations):
        for annot in annotations:
            if annot["bbox"] == []:
                return True
            for o in annot['bbox'][2:]:
                if o <= 1:
                    return True

    def _load_image(self, image_id : int):
        path = self.coco.loadImgs(image_id)[0]["file_name"]
        image = cv2.imread(os.path.join(self.image_path, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def _load_target(self, image_id: int) -> List[Any]:
        annot_ids = self.coco.getAnnIds(image_id)

        boxes = np.zeros((0, 4))
        labels = np.zeros((0, 1))

        annots = self.coco.loadAnns(annot_ids)

        for annot in annots:
            box = np.zeros((1, 4))
            label = np.zeros((1, 1))

            box[0, :4] = annot['bbox']
            label[0, :1] = self.coco_ids_to_continuous_ids[annot['category_id']]

            boxes = np.append(boxes, box, axis=0)
            labels = np.append(labels, label, axis=0)
        labels = labels.ravel()
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = np.array([annot['area'] for annot in annots], dtype=np.float32)
        iscrowd = np.array([annot['iscrowd'] for annot in annots], dtype=np.uint8)

        # Strangely, when only the mask is converted to NumPy, an error occurs.
        # It seems that albumentations does not support it.
        masks = [self.coco.annToMask(annot) for annot in annots]

        return {'image_id': torch.tensor([image_id]), 'boxes': boxes, 'masks': masks, 'labels': labels, 'area': area, 'iscrowd': iscrowd}

    def __getitem__(self, index : int) -> Tuple[Any, Any]:
        image_id = self.ids[index]
        image = self._load_image(image_id)
        target = self._load_target(image_id)
        img_meta = dict(ori_shape=image.shape[:2], image_id=image_id)

        if self.transforms is not None:
            transformed = self.transforms(image=image,
                                          masks=target['masks'],
                                          bboxes=target['boxes'],
                                          category_ids=target['labels'])
            image = transformed['image']
            img_meta['img_shape'] = image.shape[-2:]
            target['masks'] = transformed['masks']
            target['boxes'] = transformed['bboxes']

        if len(target['boxes']) == 0 or len(target['masks']) == 0 or len(target['labels']) == 0:
            # target['image_id'] = torch.as_tensor(target['image_id'], dtype=torch.int32)
            # target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            # target["masks"] = torch.zeros(
            #     0, image.shape[0], image.shape[1], dtype=torch.uint8
            # )
            # target["labels"] = torch.zeros(0, dtype=torch.int64)
            # target["area"] = torch.zeros(0, dtype=torch.int64)
            # target["iscrowd"] = torch.zeros(0, dtype=torch.int64)
            # return image, target, img_meta
            return self.__getitem__(np.random.randint(self.__len__()))

        target['image_id'] = torch.as_tensor(target['image_id'], dtype=torch.int32)
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype=torch.uint8)

        return image, target, img_meta


if __name__ == "__main__":
    import random
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2

    from matplotlib import patches
    from matplotlib import pyplot as plt

    from prettyprinter import cpprint

    # from augmentation import Augment_HSV, HistogramEqualization, Mosaic


    image_path = '/data/cocodataset/val2017'
    annFile = "/data/cocodataset/annotations/instances_val2017.json"
    dataset = COCODataset(image_path, annFile, None)

    transforms = A.Compose([
        # Mosaic(img_size=256, dataset_obj=dataset, p=1),
        A.Resize(512, 512),
        A.HorizontalFlip(p=1),
        A.Affine(p=1),
        A.Blur(p=1),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    dataset.set_transforms(transforms)
    print(f'Number of Images : {len(dataset)}')
    print(f'Number of Lables : {len(dataset.CLASSES)}')

    idx = random.randint(1, 100)
    image, target, img_metas = dataset.__getitem__(index=idx)

    if torch.is_tensor(image):
        image = image.permute(1, 2, 0).detach().numpy()
    masks = target['masks']

    print(f'Image Shape : {image.shape}')
    print(f'Mask Shapes : {masks.shape}')
    print()

    for i, (box, label) in enumerate(zip(target['boxes'], target['labels'])):
        cpprint(f'Bbox_{i + 1} : {box}')
        label = dataset.coco_ids_to_class_names[dataset.coco_ids[label.item()]]
        cpprint(f'label_{i + 1} : {label}')
        print()

    plt.imshow(image)

    for box in target['boxes']:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='red', fill=False)
        ax = plt.gca()
        ax.add_patch(rect)

    plt.show()

    for mask in masks:
        mask = np.array(mask)
        plt.imshow(mask)
        plt.show()