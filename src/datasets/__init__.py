from torch.utils.data import DataLoader

from .coco import COCODataset


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(data_root_path, image_path, annFile, transform, batch_size, num_workers, shuffle):
    image_path = data_root_path / image_path
    annFile = data_root_path / annFile

    dataset = COCODataset(image_path=image_path, annFile=annFile, transforms=None)
    dataset.set_transforms(transforms=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return loader