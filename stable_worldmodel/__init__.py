import ctypes
import ctypes.util
import os
import sys
from pathlib import Path


def _configure_headless_gl() -> None:
    """Configure PyOpenGL and MuJoCo library discovery for user-space .venv/lib."""
    import torch

    venv_lib = Path(sys.prefix) / 'lib'
    repo_venv_lib = Path(__file__).resolve().parent.parent / '.venv' / 'lib'
    search_dirs = [d for d in (venv_lib, repo_venv_lib) if d.is_dir()]

    use_osmesa = False
    if not torch.cuda.is_available():
        for d in search_dirs:
            if (d / 'libOSMesa.so.8').exists():
                use_osmesa = True
                os.environ['MUJOCO_GL'] = 'osmesa'
                os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
                break
    else:
        os.environ.setdefault('MUJOCO_GL', 'egl')

    if search_dirs:
        orig_find_library = ctypes.util.find_library

        def _patched_find_library(name: str) -> str | None:
            if use_osmesa and name in ('GL', 'OSMesa'):
                for d in search_dirs:
                    cand = d / 'libOSMesa.so.8'
                    if cand.exists():
                        return str(cand)
            for d in search_dirs:
                for candidate in (
                    d / f'lib{name}.so',
                    d / f'lib{name}.so.8',
                    d / f'lib{name}.so.6',
                    d / f'lib{name}.so.1',
                    d / f'lib{name}.so.0',
                ):
                    if candidate.exists():
                        return str(candidate)
            return orig_find_library(name)

        ctypes.util.find_library = _patched_find_library

        if use_osmesa:
            for d in search_dirs:
                for lib_name in (
                    'libedit.so.2',
                    'libglapi.so.0',
                    'libLLVM-17.so.1',
                    'libOSMesa.so.8',
                ):
                    lib_path = d / lib_name
                    if lib_path.exists():
                        try:
                            ctypes.CDLL(str(lib_path))
                        except OSError:
                            pass


_configure_headless_gl()

from stable_worldmodel import data, envs, policy, solver, spaces, utils, wrapper  # noqa: E402
from stable_worldmodel.world import World  # noqa: E402
from stable_worldmodel.policy import HierarchicalPlanConfig, PlanConfig  # noqa: E402
from stable_worldmodel.utils import pretraining  # noqa: E402

__all__ = [
    'World',
    'PlanConfig',
    'HierarchicalPlanConfig',
    'pretraining',
    'data',
    'envs',
    'policy',
    'solver',
    'spaces',
    'utils',
    'wrapper',
]
