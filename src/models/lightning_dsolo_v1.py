import os, sys
import os.path as osp
add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)


import wandb
import torch
import torch.optim as optimizer
import pytorch_lightning as pl

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from pathlib import Path
from torchinfo import summary
from omegaconf import OmegaConf

from src.modules import DSolo_v1
from src.datasets import get_dataloader
from src.eval import (
    vis_seg, get_masks,
    coco_eval, results2json, results2json_segm
)

ROOT_DIR = Path(osp.abspath(osp.join(osp.abspath(__file__),
                                osp.pardir, # models
                                osp.pardir, # modules
                                osp.pardir)))


class Lightning_DSolo_v1(pl.LightningModule):
    def __init__(self, cfgs, debug=False):
        super().__init__()
        self.cfgs = cfgs
        self.all_setup()
        self.automatic_optimization = True

        if debug:
            print("Config:\n")
            print(OmegaConf.to_yaml(main_cfg))

            print('\nModel summary:\n')
            summary(self.model)
            print('\nOptimizer:\n', self.optim)
            print('\nScheduler:\n', self.sched)
            print('\nTrain loader:\n', self.trn_dl)
            print('\nValid loader:\n', self.val_dl)


    def configure_model(self):
        self.model = DSolo_v1(self.cfgs['model'])

        if self.cfgs['model']['pretrained'] != '':
            lightning_dict = torch.load(str(ROOT_DIR / self.cfgs['model']['pretrained']), map_location='cpu')
            model_dict = lightning_dict['state_dict']
            orig_dict = self.model.state_dict()

            new_model_dict = {k[6:]:v for k, v in model_dict.items() if k[6:] in orig_dict.keys()}
            orig_dict.update(new_model_dict)
            self.model.load_state_dict(orig_dict)
            print("SynthText Pretrained loaded.\n")


    def configure_optimizers(self):
        optim_name = self.cfgs['optim']['name']
        optim_args = self.cfgs['optim']['args']
        self.optim = getattr(optimizer, optim_name)(self.parameters(), **optim_args)

        sched_cfg = self.cfgs.get('sched', {})
        if sched_cfg is None:
            self.sched = None
        else:
            sched_name = sched_cfg['name']
            sched_args = sched_cfg['args']
            self.sched = getattr(optimizer.lr_scheduler, sched_name)(self.optim, **sched_args)

        return dict(optimizer=self.optim, scheduler=self.sched)


    def configure_dataloader(self):
        data_root_dir = Path('/data') if self.cfgs['general']['data_root'] == '' \
                                else  Path(osp.join(ROOT_DIR, osp.pardir, 'data'))

        trn_size = self.cfgs['general']['trn_img_size']
        val_size = self.cfgs['general']['val_img_size']
        mean = list(self.cfgs['general']['mean'])
        std  = list(self.cfgs['general']['std'])
        train_transform = A.Compose([
            A.Resize(**trn_size),
            A.Affine(),
            A.Blur(),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

        valid_transform = A.Compose([
            A.Resize(**val_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

        self.trn_dl = get_dataloader(data_root_dir, **self.cfgs['train_loader'], transform=train_transform)
        self.val_dl = get_dataloader(data_root_dir, **self.cfgs['valid_loader'], transform=valid_transform)


    def all_setup(self):
        # wandb setup
        self.log_step = self.cfgs['experiment']['log_step']

        # model config load
        model_cfg_p = ROOT_DIR / self.cfgs['model']
        model_cfg = OmegaConf.load(model_cfg_p)
        self.cfgs['model'] = model_cfg

        # model, optimizer, dataloader setup
        self.configure_model()
        self.configure_optimizers()
        self.configure_dataloader()


    def forward(self, batch):
        return self.model(batch)


    def on_train_epoch_end(self):
        if self.sched is not None:
            self.sched.step()


    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        losses = self.model(images, targets=targets)

        # logging
        log_template = {
            "trn_total_loss": losses['loss_total'].item(),
            "trn_mask_loss": losses['loss_mask'].item(),
            "trn_cate_loss": losses['loss_cate'].item(),
            "trn_total_loss": losses['loss_total'].item(),
        }

        self.log_dict(
            log_template,
            on_step=self.log_step, on_epoch=True, prog_bar=True, logger=True
        )

        return dict(loss=losses['loss_total'])


    def on_validation_epoch_start(self):
        self.results = []


    def on_validation_epoch_end(self):
        eval_types = ['segm']
        save_result_file = str(ROOT_DIR / 'results' / 'validation_on_training')
        print('\n\nStarting evaluate {}'.format(' and '.join(eval_types)))

        if eval_types == ['proposal_fast']:
            result_file = save_result_file
            coco_eval(result_file, eval_types, self.val_dl.dataset.coco)
        else:
            if not isinstance(self.results[0], dict):
                result_files = results2json_segm(self.val_dl.dataset, self.results, save_result_file)
                coco_eval(result_files, eval_types, self.val_dl.dataset.coco)
            else:
                for name in self.results[0]:
                    print('\nEvaluating {}'.format(name))
                    outputs_ = [out[name] for out in self.results]
                    result_file = save_result_file + '.{}'.format(name)
                    result_files = results2json(self.val_dl.dataset, outputs_, result_file)
                    coco_eval(result_files, eval_types, self.val_dl.dataset.coco)


    def validation_step(self, batch, batch_idx):
        images, targets, img_metas = batch
        outputs = self.model(images, img_metas=img_metas)

        if batch_idx == 0:
            vis_results = vis_seg(images, img_metas, outputs,
                                score_thr=self.cfgs['model']['post_process_args']['score_thr'],
                                mask_thr=self.cfgs['model']['post_process_args']['score_thr'],
                                save_dir=str(ROOT_DIR / 'debug' / 'valid' / str(self.current_epoch).zfill(2)))

            wandb.log({"Results of Validation": [wandb.Image(log_image) for log_image in vis_results]})

        result = get_masks(outputs, num_classes=len(self.trn_dl.dataset))
        self.results.append(result)


if __name__ == "__main__":
    from pytorch_lightning import seed_everything

    cfg_path = ROOT_DIR / 'configs' / 'experiments' / 'light_dsolo_r50_fpn_3x.yaml'
    print(cfg_path)

    main_cfg = OmegaConf.load(cfg_path)
    seed_everything(main_cfg['general']['seed'])
    model = Lightning_DSolo_v1(main_cfg, debug=True)