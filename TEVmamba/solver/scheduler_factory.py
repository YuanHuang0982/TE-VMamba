""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler

def create_scheduler(cfg, optimizer):
    print("BASE_LR type:", type(cfg.SOLVER.BASE_LR), cfg.SOLVER.BASE_LR)

    num_epochs = cfg.SOLVER.MAX_EPOCHS
    warmup_t = cfg.SOLVER.WARMUP_EPOCHS

    # 그룹별 초기 lr 읽기
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    # 기존 비율 그대로 유지
    lr_min_ratio = 0.002
    warmup_ratio = 0.01

    # 그룹별로 lr_min / warmup_lr_init 생성
    lr_min = [lr_min_ratio * lr for lr in base_lrs]
    warmup_lr_init = [warmup_ratio * lr for lr in base_lrs]

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=lr_min,
        t_mul=1.,
        decay_rate=0.1,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.,
        noise_seed=42,
    )
    return lr_scheduler
