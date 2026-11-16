import os

if os.environ.get('MPLBACKEND', '').startswith('module://'):
    os.environ['MPLBACKEND'] = 'Agg'

from .models import (
    DiffusionWorldModel,
    SDVAEWrapper,
    SpatioTemporalDiT,
    TinyConvVAE,
    build_diffusion_world_model,
    build_vae,
)

__all__ = [
    'DiffusionWorldModel',
    'SDVAEWrapper',
    'SpatioTemporalDiT',
    'TinyConvVAE',
    'build_diffusion_world_model',
    'build_vae',
]
