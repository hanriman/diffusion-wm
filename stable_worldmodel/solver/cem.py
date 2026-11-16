"""Cross-Entropy Method (CEM) and MPPI solver for model-based planning."""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .solver import Costable, get_planning_horizon


def _resolve_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        return torch.device('cpu')
    return dev


class CEMSolver(torch.nn.Module):
    """Sampling-based Cross-Entropy Method (CEM) / MPPI solver.

    Iteratively samples candidate action sequences from a factorized Gaussian
    distribution, evaluates world-model rollout costs under ``torch.no_grad()``,
    and updates the Gaussian mean and standard deviation using the top-k elite
    candidates (or MPPI soft-exponential weights when ``temperature > 0``).

    Args:
        model: World model implementing the ``Costable`` protocol.
        n_steps: Number of CEM refinement iterations.
        num_samples: Number of candidate action sequences sampled per iteration.
        num_elites: Number of top candidates used to update distribution parameters.
        var_scale: Initial standard deviation scale for action sampling.
        min_std: Minimum standard deviation floor during CEM updates.
        momentum: Momentum factor ``alpha`` in ``[0, 1)`` for smoothing mean/std updates
            (``new = (1 - alpha) * elite_stat + alpha * old``).
        temperature: Optional MPPI temperature ``> 0``. If ``None`` or ``<= 0``,
            uses uniform elite averaging (standard CEM).
        action_clip_sigma: Clip sampled actions to ``[-sigma, sigma]`` in normalized space.
        batch_size: Number of environments to process in parallel per batch.
        device: Computation device ('cuda' or 'cpu').
        seed: Random seed for reproducible sampling.
    """

    def __init__(
        self,
        model: Costable,
        n_steps: int = 5,
        num_samples: int = 256,
        num_elites: int = 32,
        var_scale: float = 1.0,
        min_std: float = 0.05,
        momentum: float = 0.1,
        temperature: float | None = None,
        action_clip_sigma: float | None = 3.0,
        batch_size: int | None = None,
        device: str | torch.device = 'cuda',
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.model = model
        self.n_steps = max(1, int(n_steps))
        self.num_samples = max(1, int(num_samples))
        self.num_elites = max(1, min(int(num_elites), self.num_samples))
        self.var_scale = float(var_scale)
        self.min_std = float(min_std)
        self.momentum = float(momentum)
        self.temperature = None if temperature is None else float(temperature)
        self.action_clip_sigma = action_clip_sigma
        self.batch_size = batch_size
        self.device = _resolve_device(device)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(int(seed))

        self._n_envs: int | None = None
        self._action_dim: int | None = None
        self._config: Any = None
        self.level: int = 1
        self.solve_records: list[dict[str, Any]] = []
        self._prev_mean: torch.Tensor | None = None

    def configure(
        self,
        *,
        action_space: gym.Space,
        n_envs: int,
        config: Any,
        level: int = 1,
    ) -> None:
        """Configure the solver with environment and planning specifications."""
        self._action_space = action_space
        self._n_envs = int(n_envs)
        self._config = config
        self.level = int(level)
        if level == 1:
            self._action_dim = int(np.prod(action_space.shape[1:]))
        else:
            self._action_dim = int(config.action_dim)
        self._prev_mean = None

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def _expand_info_dict_for_samples(
        self,
        info_dict: dict,
        start_idx: int,
        end_idx: int,
        num_samples: int,
    ) -> dict:
        current_bs = end_idx - start_idx
        expanded_infos = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                v_batch = v[start_idx:end_idx].to(self.device)
                v_batch = v_batch.unsqueeze(1).expand(
                    current_bs, num_samples, *v_batch.shape[1:]
                )
            elif isinstance(v, np.ndarray):
                v_batch = v[start_idx:end_idx]
                v_batch = np.repeat(v_batch[:, None, ...], num_samples, axis=1)
            else:
                v_batch = v
            expanded_infos[k] = v_batch
        return expanded_infos

    @torch.inference_mode()
    def _rollout_final_actions(self, info_dict: dict, actions: torch.Tensor) -> dict:
        action_candidates = actions.to(self.device).unsqueeze(1)
        expanded = self._expand_info_dict_for_samples(
            info_dict, start_idx=0, end_idx=self.n_envs, num_samples=1
        )
        rollout_info = self.model.rollout(expanded, action_candidates)
        predictions: dict[str, torch.Tensor] = {}
        for k, v in rollout_info.items():
            if not (k.startswith('predicted') and torch.is_tensor(v)):
                continue
            value = v.detach().cpu()
            if value.ndim > 1 and value.shape[1] == 1:
                value = value[:, 0]
            predictions[k] = value
        return predictions

    def _get_action_prior_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        queue = getattr(self.model, 'get_latent_action_queue', lambda: None)()
        if torch.is_tensor(queue) and queue.numel() > 0 and queue.ndim == 2:
            queue = queue.detach().to(self.device)
            mean = queue.mean(dim=0)
            std = queue.std(dim=0, unbiased=False).clamp_min(1e-4)
            return mean, std
        mean = torch.zeros(self.action_dim, device=self.device)
        std = torch.ones(self.action_dim, device=self.device)
        return mean, std

    @torch.no_grad()
    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
        planning_horizon: int | None = None,
        steps_taken: int | None = None,
        eval_budget: int | None = None,
    ) -> dict:
        """Solve the planning problem using CEM / MPPI."""
        start_time = time.time()
        cost_last_n = max(1, int(getattr(self._config, 'cost_last_n', 1)))
        intermediate_cost_weight = float(
            getattr(self._config, 'intermediate_cost_weight', 1.0)
        )
        if planning_horizon is None:
            planning_horizon = get_planning_horizon(
                configured_horizon=self.horizon,
                replanning_interval=int(self._config.receding_horizon)
                * int(self._config.action_block),
                steps_taken=steps_taken,
                eval_budget=eval_budget,
            )

        prior_mean, prior_std = self._get_action_prior_stats()
        base_std = prior_std * self.var_scale

        if init_action is not None:
            mean = init_action.to(self.device)
            if mean.shape[1] < planning_horizon:
                pad = prior_mean.view(1, 1, -1).expand(
                    self.n_envs, planning_horizon - mean.shape[1], -1
                )
                mean = torch.cat([mean, pad], dim=1)
            else:
                mean = mean[:, :planning_horizon].clone()
        else:
            mean = prior_mean.view(1, 1, -1).expand(
                self.n_envs, planning_horizon, -1
            ).clone()

        std = base_std.view(1, 1, -1).expand(
            self.n_envs, planning_horizon, -1
        ).clone()

        batch_size = (
            self.batch_size if self.batch_size is not None else self.n_envs
        )
        total_envs = self.n_envs
        batch_top_actions_list = []
        per_env_iters = []
        per_env_time = []

        for start_idx in range(0, total_envs, batch_size):
            end_idx = min(start_idx + batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_t0 = time.time()

            b_mean = mean[start_idx:end_idx].clone()
            b_std = std[start_idx:end_idx].clone()

            expanded_infos = self._expand_info_dict_for_samples(
                info_dict, start_idx, end_idx, num_samples=self.num_samples
            )

            best_actions = b_mean.clone()
            best_costs = torch.full(
                (current_bs,), float('inf'), device=self.device
            )

            for _ in range(self.n_steps):
                noise = torch.randn(
                    (current_bs, self.num_samples, planning_horizon, self.action_dim),
                    generator=self.torch_gen,
                    device=self.device,
                )
                candidates = b_mean.unsqueeze(1) + b_std.unsqueeze(1) * noise
                # Include current mean as candidate 0 for elitist stability
                candidates[:, 0] = b_mean

                if self.action_clip_sigma is not None:
                    lower = (prior_mean - self.action_clip_sigma * prior_std).view(1, 1, 1, -1)
                    upper = (prior_mean + self.action_clip_sigma * prior_std).view(1, 1, 1, -1)
                    candidates.clamp_(min=lower, max=upper)

                current_info = expanded_infos.copy()
                current_info['cost_last_n'] = cost_last_n
                current_info['intermediate_cost_weight'] = intermediate_cost_weight

                costs = self.model.get_cost(current_info, candidates)
                for k in ('embed_0', 'goal_embed_0', 'proprio_0', 'goal_proprio_0'):
                    if k in current_info:
                        expanded_infos.setdefault(k, current_info[k])

                elite_costs, elite_idx = torch.topk(
                    costs, k=self.num_elites, dim=1, largest=False
                )
                idx_expanded = elite_idx.unsqueeze(-1).unsqueeze(-1).expand(
                    current_bs, self.num_elites, planning_horizon, self.action_dim
                )
                elites = torch.gather(candidates, dim=1, index=idx_expanded)

                # Track best candidate overall
                iter_best_cost = elite_costs[:, 0]
                improved = iter_best_cost < best_costs
                best_costs = torch.where(improved, iter_best_cost, best_costs)
                best_actions = torch.where(
                    improved.view(-1, 1, 1), elites[:, 0], best_actions
                )

                if self.temperature is not None and self.temperature > 0:
                    # MPPI soft weighting over elites
                    shifted = elite_costs - elite_costs.min(dim=1, keepdim=True).values
                    weights = torch.softmax(-shifted / self.temperature, dim=1)
                    w_view = weights.view(current_bs, self.num_elites, 1, 1)
                    new_mean = (w_view * elites).sum(dim=1)
                    new_var = (w_view * (elites - new_mean.unsqueeze(1)).pow(2)).sum(dim=1)
                    new_std = torch.sqrt(new_var + 1e-8).clamp_min(self.min_std)
                else:
                    new_mean = elites.mean(dim=1)
                    new_std = elites.std(dim=1, unbiased=False).clamp_min(self.min_std)

                b_mean = (1.0 - self.momentum) * new_mean + self.momentum * b_mean
                b_std = (1.0 - self.momentum) * new_std + self.momentum * b_std

            batch_top_actions_list.append(best_actions.detach().cpu())
            per_env_iters.extend([self.n_steps] * current_bs)
            per_env_time.extend([time.time() - batch_t0] * current_bs)

        outputs = {}
        outputs['actions'] = torch.cat(batch_top_actions_list, dim=0)
        outputs['predictions'] = self._rollout_final_actions(
            info_dict, outputs['actions']
        )
        end_time = time.time()

        self.solve_records.append({
            'level': int(self.level),
            'horizon': int(planning_horizon),
            'num_samples': int(self.num_samples),
            'n_iters': per_env_iters,
            'wall_clock': end_time - start_time,
            'wall_clock_per_env': per_env_time,
        })
        return outputs
