""" Cosine Scheduler

Cosine LR schedule with warmup, cycle/restarts, noise.

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import math
import torch

from .scheduler import Scheduler


_logger = logging.getLogger(__name__)


class CosineLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        assert t_initial > 0

        # --- lr_min: scalar or list ---
        if isinstance(lr_min, (list, tuple)):
            assert len(lr_min) == len(self.base_values)
            assert all(v >= 0 for v in lr_min)
            self.lr_min = [float(v) for v in lr_min]
        else:
            assert float(lr_min) >= 0
            self.lr_min = float(lr_min)

        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            _logger.warning("Cosine annealing scheduler will have no effect on the learning "
                            "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = int(warmup_t)
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs

        # --- warmup_lr_init: scalar or list ---
        if isinstance(warmup_lr_init, (list, tuple)):
            assert len(warmup_lr_init) == len(self.base_values)
            self.warmup_lr_init = [float(v) for v in warmup_lr_init]
        else:
            self.warmup_lr_init = float(warmup_lr_init)

        if self.warmup_t:
            if isinstance(self.warmup_lr_init, list):
                self.warmup_steps = [(b - w) / self.warmup_t for b, w in zip(self.base_values, self.warmup_lr_init)]
                super().update_groups(self.warmup_lr_init)  # list OK
            else:
                self.warmup_steps = [(b - self.warmup_lr_init) / self.warmup_t for b in self.base_values]
                super().update_groups(self.warmup_lr_init)  # scalar OK
        else:
            self.warmup_steps = [1.0 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            if isinstance(self.warmup_lr_init, list):
                lrs = [w + t * s for w, s in zip(self.warmup_lr_init, self.warmup_steps)]
            else:
                lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.decay_rate ** i
            lr_max_values = [v * gamma for v in self.base_values]

            # lr_min values (scalar or per-group)
            if isinstance(self.lr_min, list):
                lr_min_values = [v * gamma for v in self.lr_min]
            else:
                lr_min_values = [self.lr_min * gamma for _ in lr_max_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min_v + 0.5 * (lr_max - lr_min_v) * (1 + math.cos(math.pi * t_curr / t_i))
                    for lr_max, lr_min_v in zip(lr_max_values, lr_min_values)
                ]
            else:
                # cycle 끝나면 lr_min으로 고정
                lrs = lr_min_values

        return lrs


    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles=0):
        if not cycles:
            cycles = self.cycle_limit
        cycles = max(1, cycles)
        if self.t_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.t_mul ** cycles - 1) / (1 - self.t_mul)))
