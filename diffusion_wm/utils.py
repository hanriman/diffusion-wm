"""Checkpoint resolution and PyTorch Lightning callbacks for Diffusion World Model."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def resolve_model_checkpoint_path(
    run_name: str | Path, cache_dir: str | Path | None = None
) -> Path:
    """Resolve a run name or path to an on-disk `*_object.ckpt` path."""
    run_path = Path(run_name).expanduser()
    if run_path.suffix == '.ckpt':
        if run_path.exists():
            return run_path
        if not run_path.is_absolute():
            import stable_worldmodel as swm

            cached_path = Path(
                cache_dir or swm.data.utils.get_cache_dir(), run_name
            )
            if cached_path.exists():
                return cached_path
        raise FileNotFoundError(
            f'Checkpoint path does not exist: {run_path}. Launch training first.'
        )

    if not run_path.exists():
        import stable_worldmodel as swm

        run_path = Path(cache_dir or swm.data.utils.get_cache_dir(), run_name)

    if run_path.is_dir():
        ckpt_files = list(run_path.glob('*_object.ckpt'))
        ckpt_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        if not ckpt_files:
            raise FileNotFoundError(
                f'No *_object.ckpt found in directory: {run_path}'
            )
        return ckpt_files[0]

    path = Path(f'{run_path}_object.ckpt')
    if not path.exists():
        raise FileNotFoundError(
            f'Checkpoint path does not exist: {path}. Launch training first.'
        )
    return path


class ModelObjectCallback(Callback):
    """Save the `DiffusionWorldModel` object (`*_object.ckpt`) periodically and at train end."""

    def __init__(
        self, dirpath: str | Path, filename: str = 'model', epoch_interval: int = 1
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = max(1, int(epoch_interval))
        self._saved_final = False

    def _save_final(self, trainer, pl_module) -> None:
        if self._saved_final or not trainer.is_global_zero:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        path = self.dirpath / f'{self.filename}_object.ckpt'
        torch.save(pl_module.model, path)
        self._saved_final = True
        logger.info(f'Saved final DiffusionWorldModel to {path}')

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0:
            path = self.dirpath / f'{self.filename}_epoch_{epoch}_object.ckpt'
            torch.save(pl_module.model, path)
            logger.info(f'Saved DiffusionWorldModel to {path}')
        if epoch == trainer.max_epochs:
            self._save_final(trainer, pl_module)

    def on_train_end(self, trainer, pl_module) -> None:
        self._save_final(trainer, pl_module)


class PlanningEvalCallback(Callback):
    """Run MPC planning evaluation at the end of training or every N epochs."""

    def __init__(
        self,
        *,
        enabled: bool,
        eval_cfg,
        run_dir: str | Path,
        seed: int,
        output_subdir: str = 'planning_eval',
        run_on_train_end: bool = True,
        every_n_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.eval_cfg = eval_cfg
        self.run_dir = Path(run_dir)
        self.seed = int(seed)
        self.output_subdir = output_subdir
        self.run_on_train_end = bool(run_on_train_end)
        self.every_n_epochs = int(every_n_epochs)
        self._last_eval_step = None

    def _run_eval(self, trainer, pl_module, epoch: int) -> None:
        from .planning_eval import load_eval_config, run_planning_eval

        cfg = load_eval_config(config_name=self.eval_cfg.config_name)
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        overrides = self.eval_cfg.get('overrides', None)
        if overrides is not None:
            cfg = OmegaConf.merge(cfg, overrides)
        cfg.solver.seed = int(self.seed)

        was_training = pl_module.model.training
        try:
            pl_module.model.eval()
            metrics = run_planning_eval(
                cfg,
                model=pl_module.model,
                results_dir=self.run_dir / self.output_subdir / f'epoch_{epoch:04d}',
            )
        finally:
            pl_module.model.train(was_training)

        if trainer.logger is not None and isinstance(metrics, dict):
            prefix = f'planning_eval/{self.eval_cfg.config_name}'
            scalars = {
                f'{prefix}/{k}': float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float, bool))
            }
            if scalars:
                trainer.logger.log_metrics(scalars, step=trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if (
            not self.enabled
            or not self.every_n_epochs
            or epoch % self.every_n_epochs != 0
        ):
            return
        if trainer.is_global_zero:
            self._run_eval(trainer, pl_module, epoch)
        self._last_eval_step = trainer.global_step

    def on_train_end(self, trainer, pl_module) -> None:
        if not self.enabled or not self.run_on_train_end or trainer.global_step <= 0:
            return
        if self._last_eval_step == trainer.global_step:
            return
        if trainer.is_global_zero:
            self._run_eval(trainer, pl_module, trainer.current_epoch + 1)
