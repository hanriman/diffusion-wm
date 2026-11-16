"""PyTorch Lightning + Hydra training entry point for Diffusion World Model."""

from __future__ import annotations

import math
import os
from pathlib import Path

import hydra
import lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    from .data import (
        Compose,
        build_hdf5_dataset,
        build_normalizer_artifact,
        get_column_normalizer,
        normalize_uint8_images_on_device,
        save_normalizer_artifact,
    )
    from .models import build_diffusion_world_model
    from .utils import ModelObjectCallback, PlanningEvalCallback
except ImportError:
    from data import (
        Compose,
        build_hdf5_dataset,
        build_normalizer_artifact,
        get_column_normalizer,
        normalize_uint8_images_on_device,
        save_normalizer_artifact,
    )
    from models import build_diffusion_world_model
    from utils import ModelObjectCallback, PlanningEvalCallback


class DiffusionWorldModelLightningModule(pl.LightningModule):
    """LightningModule wrapping `DiffusionWorldModel` for training and validation."""

    def __init__(self, model: torch.nn.Module, cfg: DictConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        batch = normalize_uint8_images_on_device(batch)
        losses = self.model.compute_loss(batch)
        bs = batch['action_level1'].size(0)
        for k, v in losses.items():
            self.log(
                f'train/{k}',
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=(k == 'loss'),
                batch_size=bs,
                sync_dist=True,
            )
        return losses['loss']

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        batch = normalize_uint8_images_on_device(batch)
        losses = self.model.compute_loss(batch)
        bs = batch['action_level1'].size(0)
        for k, v in losses.items():
            self.log(
                f'val/{k}',
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=(k == 'loss'),
                batch_size=bs,
                sync_dist=True,
            )

        if batch_idx == 0:
            actions = batch['action_level1'].float()
            if 'vae_latents_level1' in batch:
                z_vis = batch['vae_latents_level1'].float()
            else:
                z_vis = self.model.vae.encode(batch['pixels_level1'])
            h_ctx = min(max(1, self.model.history_size), z_vis.size(1) - 1)
            n_fut = z_vis.size(1) - h_ctx
            if n_fut > 0:
                ctx_vis = z_vis[:, :h_ctx]
                fut_vis_gt = z_vis[:, h_ctx:]
                # Actions transitioning from frame (h_ctx - 1) .. (T - 2) into frames h_ctx .. (T - 1)
                fut_actions = actions[:, h_ctx - 1 : h_ctx - 1 + n_fut]
                ctx_p = (
                    batch['proprio_level1'][:, :h_ctx].float()
                    if self.model.use_proprio
                    else None
                )
                pred_vis_all, pred_p_all = self.model.sample_ode_parallel(
                    ctx_vis, fut_actions, ctx_proprio=ctx_p
                )
                ode_vis_mse = F.mse_loss(pred_vis_all[:, h_ctx:], fut_vis_gt)
                self.log(
                    'val/ode_vis_mse',
                    ode_vis_mse,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=bs,
                    sync_dist=True,
                )
                if self.model.use_proprio and pred_p_all is not None:
                    fut_p_gt = batch['proprio_level1'][:, h_ctx:].float()
                    ode_p_mse = F.mse_loss(pred_p_all[:, h_ctx:], fut_p_gt)
                    self.log(
                        'val/ode_proprio_mse',
                        ode_p_mse,
                        on_step=False,
                        on_epoch=True,
                        batch_size=bs,
                        sync_dist=True,
                    )

        return losses['loss']

    def configure_optimizers(self):
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        opt_cfg = self.cfg.optimizer
        lr = float(opt_cfg.get('lr', 1e-4))
        weight_decay = float(opt_cfg.get('weight_decay', 1e-3))
        optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay
        )

        warmup_ratio = float(opt_cfg.get('warmup_ratio', 0.05))
        total_steps = int(getattr(self.trainer, 'estimated_stepping_batches', 0))
        if total_steps <= 0 or warmup_ratio <= 0:
            return optimizer

        warmup_steps = max(1, int(total_steps * warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }


def _configure_runtime_performance() -> None:
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def train_from_config(cfg: DictConfig) -> DiffusionWorldModelLightningModule:
    _configure_runtime_performance()
    pl.seed_everything(int(cfg.seed), workers=True)

    cache_dir = cfg.get('cache_dir', None) or os.environ.get('HJEPA_HOME', None)
    if cache_dir is None:
        cache_dir = str(Path.home() / '.stable_worldmodel')

    dataset_cfg = {
        k: v for k, v in cfg.data.dataset.items() if k != 'val_name'
    }
    val_total_transitions = dataset_cfg.pop('val_total_transitions', None)
    val_name = cfg.data.dataset.get('val_name', None) or dataset_cfg['name']

    train_dataset = build_hdf5_dataset(dataset_cfg, cache_dir=cache_dir)

    val_dataset_cfg = {
        k: v
        for k, v in dataset_cfg.items()
        if k not in ('total_transitions', 'data_path')
    }
    val_dataset_cfg['name'] = val_name
    if val_total_transitions is not None:
        val_dataset_cfg['total_transitions'] = int(val_total_transitions)
    val_dataset = build_hdf5_dataset(val_dataset_cfg, cache_dir=cache_dir)

    extra_transforms = [
        get_column_normalizer(train_dataset, col, col)
        for col in cfg.data.dataset.keys_to_load
        if not col.startswith('pixels') and not col.startswith('vae_latents')
    ]
    transform = Compose(*extra_transforms)
    train_dataset.transform = transform
    val_dataset.transform = transform

    loader_cfg = OmegaConf.to_container(cfg.loader, resolve=True)
    if int(loader_cfg.get('num_workers', 0)) == 0:
        loader_cfg.pop('persistent_workers', None)
        loader_cfg.pop('prefetch_factor', None)
    if not torch.cuda.is_available():
        loader_cfg['pin_memory'] = False

    rnd_gen = torch.Generator().manual_seed(int(cfg.seed))
    train_loader = DataLoader(
        train_dataset,
        **loader_cfg,
        generator=rnd_gen,
    )
    val_loader_cfg = dict(loader_cfg)
    val_loader_cfg['shuffle'] = False
    val_loader_cfg['drop_last'] = False
    val_loader = DataLoader(val_dataset, **val_loader_cfg)

    normalizer_artifact = build_normalizer_artifact(cfg, train_dataset)
    model = build_diffusion_world_model(cfg)
    model.normalizer_artifact = normalizer_artifact

    run_dir = Path(cfg.get('run_dir', None) or (Path(cache_dir) / 'ckpts' / cfg.subdir))
    run_dir.mkdir(parents=True, exist_ok=True)
    save_normalizer_artifact(normalizer_artifact, run_dir)
    (run_dir / 'config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    logging.info(f'Saved training config and normalizer.pt to {run_dir}')

    callbacks = []
    if cfg.get('dump_object', True):
        callbacks.append(
            ModelObjectCallback(
                dirpath=run_dir,
                filename=str(cfg.output_model_name),
                epoch_interval=int(cfg.get('save_every_n_epochs', 1)),
            )
        )

    planning_eval_cfg = cfg.get('planning_eval', None)
    if planning_eval_cfg is not None:
        callbacks.append(
            PlanningEvalCallback(
                enabled=bool(planning_eval_cfg.get('enabled', False)),
                eval_cfg=planning_eval_cfg,
                run_dir=run_dir,
                seed=int(cfg.seed),
                output_subdir=str(
                    planning_eval_cfg.get('output_subdir', 'planning_eval')
                ),
                run_on_train_end=bool(
                    planning_eval_cfg.get('run_on_train_end', True)
                ),
                every_n_epochs=int(planning_eval_cfg.get('every_n_epochs', 0)),
            )
        )

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    if not torch.cuda.is_available():
        trainer_cfg['accelerator'] = 'cpu'
        trainer_cfg['devices'] = 1
        trainer_cfg['precision'] = '32-true'

    pl_module = DiffusionWorldModelLightningModule(model=model, cfg=cfg)
    trainer = pl.Trainer(
        **trainer_cfg,
        default_root_dir=str(run_dir),
        callbacks=callbacks,
        logger=bool(cfg.get('enable_logger', True)),
        enable_checkpointing=False,
    )
    trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return pl_module


@hydra.main(version_base=None, config_path='config/train', config_name='pusht_dwm')
def run(cfg: DictConfig) -> None:
    train_from_config(cfg)


if __name__ == '__main__':
    run()
