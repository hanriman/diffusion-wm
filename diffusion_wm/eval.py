"""Hydra CLI entry point for standalone planning evaluation."""

import os

if os.environ.get('MPLBACKEND', '').startswith('module://'):
    os.environ['MPLBACKEND'] = 'Agg'

import hydra
from omegaconf import DictConfig

try:
    from .planning_eval import run_planning_eval
except ImportError:
    from planning_eval import run_planning_eval


@hydra.main(version_base=None, config_path='config/eval', config_name='pusht_gd')
def run(cfg: DictConfig) -> None:
    run_planning_eval(cfg)


if __name__ == '__main__':
    run()
