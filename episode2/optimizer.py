import math

import torch


def cosine_decay(lr: float, min_lr: float, step: int, decay_steps: int) -> float:
    t = step / decay_steps
    w = (1.0 + math.cos(math.pi * t)) * 0.5
    return min_lr + (lr - min_lr) * w


class LRScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 0,
        max_decay_steps: int = 0,
        cosine_decay: bool = False,
        min_lr: float = 0.0,
    ):
        self.cur_step = 0
        self.warmup_steps = warmup_steps
        self.max_decay_steps = max_decay_steps
        self.cosine_decay = cosine_decay
        self.min_lr = min_lr
        super().__init__(optimizer)

    def step(self, step: int = None) -> None:
        self.cur_step = step if step is not None else self.cur_step + 1
        super().step()

    def get_lr(self) -> list[float]:
        return [self._get_lr(base_lr) for base_lr in self.base_lrs]

    def _get_lr(self, base_lr: float) -> float:
        step = float(self.cur_step)
        max_step = float(self.max_decay_steps)

        lr = base_lr

        if self.warmup_steps:
            lr *= min(1.0, (step + 1.0) / self.warmup_steps)
            step = max(0.0, step - self.warmup_steps)
            max_step = max(0.0, max_step - self.warmup_steps)

        if step == 0:
            return lr

        if max_step:
            step = min(step, max_step)

        if self.cosine_decay:
            lr = cosine_decay(lr, self.min_lr, step, decay_steps=max_step)

        if self.min_lr:
            lr = max(lr, self.min_lr)

        return lr
