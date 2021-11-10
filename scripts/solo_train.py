import os.path as osp
import sys
add_dir = osp.dirname(osp.abspath(osp.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{osp.sep}'.join(add_dir.split(osp.sep)[:-1])
sys.path.append(add_dir)


import wandb
import hydra
from omegaconf import ListConfig

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from src.models import Lightning_Solo_v1


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))


# @hydra.main(config_path='../configs/experiments', config_name='light_dsolo_r50_fpn_3x')
@hydra.main(config_path='../configs')
def main(cfgs):
    if 'experiments' in cfgs:
        cfgs = cfgs['experiments']

    seed_everything(cfgs['general']['seed'])
    if cfgs['experiment']['wandb']:
        wandb_logger = WandbLogger(
                                project='[Instance Segmentation]',
                                name=cfgs['experiment']['name'],
                                log_model=cfgs['experiment']['log_model'],
                                config=cfgs,
                                group=cfgs['experiment']['group'])
    else:
        wandb_logger = None


    trainer_cfg = {
                   'gpus': cfgs['general']['gpus'],
                   'logger': wandb_logger,
                   'log_every_n_steps': 10,
                   'deterministic': True,
                   'precision': cfgs['general']['precision']}

    trainer_cfg['val_check_interval'] = 1
    trainer_cfg['num_sanity_val_steps'] = 10
    trainer_cfg['max_epochs'] = cfgs['general']['epoch']

    if cfgs['grad_clip']['use']:
        trainer_cfg['gradient_clip_val'] = cfgs['grad_clip']['max_norm']


    if isinstance(cfgs['general']['gpus'], ListConfig) and len(cfgs['general']['gpus']) > 1:
        trainer_cfg['num_nodes'] = 1
        trainer_cfg['accelerator'] = 'ddp'
        trainer_cfg['plugins'] = DDPPlugin(find_unused_parameters=True)

    checkpoint_dict = {
        'dirpath': f"{ROOT_DIR}/ckpts/{cfgs['experiment']['name']}",
        'save_last': True,
        'auto_insert_metric_name': True
    }


    checkpoint_dict['monitor'] = 'loss'
    checkpoint_dict['filename'] = f"{cfgs['experiment']['name']}-" + "{loss:.6f}"
    checkpoint_dict['save_top_k'] = cfgs['general']['save_top_k']
    checkpoint_dict['every_n_epochs'] = cfgs['general']['save_interval']
    checkpoint_dict['mode'] = 'min'
    checkpoint_callback = ModelCheckpoint(**checkpoint_dict)

    pl_model = Lightning_Solo_v1(cfgs, debug=False)
    pl_trainer = Trainer(**trainer_cfg, callbacks=[checkpoint_callback])
    if wandb_logger is not None:
        wandb_logger.watch(pl_model)

    fit_args = {
        'model': pl_model,
        'train_dataloaders': pl_model.trn_dl,
        'val_dataloaders': pl_model.val_dl
    }
    pl_trainer.fit(**fit_args)
    wandb.finish()


if __name__ == '__main__':
    main()