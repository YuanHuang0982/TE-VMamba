import torch
import torch.nn as nn

def make_optimizer(cfg, model, center_criterion):
    base_lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    wd_bias = cfg.SOLVER.WEIGHT_DECAY_BIAS

    # -----------------------------
    # 1) no_weight_decay / keywords 수집
    # -----------------------------
    nwd_names = set()
    nwd_keywords = set()

    # wrapper(model) 또는 내부 backbone(model.base) 어느 쪽에 있든 흡수
    for obj in [model, getattr(model, "base", None)]:
        if obj is None:
            continue
        if hasattr(obj, "no_weight_decay") and callable(getattr(obj, "no_weight_decay")):
            try:
                nwd_names |= set(obj.no_weight_decay())
            except Exception:
                pass
        if hasattr(obj, "no_weight_decay_keywords") and callable(getattr(obj, "no_weight_decay_keywords")):
            try:
                nwd_keywords |= set(obj.no_weight_decay_keywords())
            except Exception:
                pass

    def is_no_weight_decay(name: str) -> bool:
        # (a) 정확히 이름 매칭
        if name in nwd_names:
            return True
        # (b) 키워드 포함
        for kw in nwd_keywords:
            if kw and (kw in name):
                return True
        # (c) 흔한 norm/bias 관례
        if name.endswith(".bias"):
            return True
        if any(x in name.lower() for x in ["norm", "bn", "ln", "layernorm", "batchnorm"]):
            return True
        return False

    # -----------------------------
    # 2) 파라미터 그룹 구성 (decay / no_decay)
    # -----------------------------
    decay_params = []
    nodecay_params = []
    decay_params_lr = []      # (옵션) FC만 lr 다르게 줄 경우 대비
    nodecay_params_lr = []

    # LARGE_FC_LR 처리용: classifier/metric head에 lr*2
    def is_fc(name: str) -> bool:
        # TransReID 관례 + 네 wrapper 관례
        return ("classifier" in name) or ("arcface" in name) or ("cosface" in name) or ("amsoftmax" in name) or ("circle" in name)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # lr 설정
        lr = base_lr
        if cfg.SOLVER.LARGE_FC_LR and is_fc(name):
            lr = base_lr * 2

        # decay / no_decay 분리
        if is_no_weight_decay(name):
            # no_decay
            if lr != base_lr:
                nodecay_params_lr.append(p)
            else:
                nodecay_params.append(p)
        else:
            # decay
            if lr != base_lr:
                decay_params_lr.append(p)
            else:
                decay_params.append(p)

    param_groups = []
    if len(decay_params) > 0:
        param_groups.append({"params": decay_params, "lr": base_lr, "weight_decay": wd})
    if len(nodecay_params) > 0:
        param_groups.append({"params": nodecay_params, "lr": base_lr, "weight_decay": 0.0})
    if len(decay_params_lr) > 0:
        param_groups.append({"params": decay_params_lr, "lr": base_lr * 2, "weight_decay": wd})
    if len(nodecay_params_lr) > 0:
        param_groups.append({"params": nodecay_params_lr, "lr": base_lr * 2, "weight_decay": 0.0})

    # -----------------------------
    # 3) Optimizer 생성
    # -----------------------------
    opt_name = cfg.SOLVER.OPTIMIZER_NAME
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(param_groups, momentum=cfg.SOLVER.MOMENTUM)
    elif opt_name == "AdamW":
        # group별 weight_decay를 이미 넣었으니, 여기 weight_decay는 0으로 두는 게 깔끔
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=0.0)
    elif opt_name == "Adam":
        optimizer = torch.optim.Adam(param_groups, lr=base_lr)
    else:
        optimizer = getattr(torch.optim, opt_name)(param_groups)

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center
