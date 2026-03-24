from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg
import config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    print("CONFIG FILE:", config.__file__)
    print("BEFORE MERGE DEVICE_ID:", cfg.MODEL.DEVICE_ID, type(cfg.MODEL.DEVICE_ID))
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)


    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # ===================== [DEBUG] prune/merge 부착 + cached_delta 체크 =====================
    def _get_backbone(model):
        m = model.module if hasattr(model, "module") else model
        for name in ["base", "backbone", "model", "net"]:
            if hasattr(m, name):
                return getattr(m, name)
        return m

    def _iter_blocks(backbone):
        # backbone.layers[*].blocks[*] 형태를 가정 
        if not hasattr(backbone, "layers"):
            return
        for li, layer in enumerate(backbone.layers):
            if not hasattr(layer, "blocks"):
                continue
            for bi, blk in enumerate(layer.blocks):
                yield li, bi, blk

    def debug_print_attach(backbone, max_print=20):
        total = 0
        n_merge = 0
        n_prune = 0
        printed = 0

        for li, bi, blk in _iter_blocks(backbone):
            total += 1
            op = getattr(blk, "op", None)
            has_merge = False
            has_prune = False

            if op is not None:
                has_merge = hasattr(op, "merge_module") and (getattr(op, "merge_module") is not None)
                has_prune = hasattr(op, "prune_module") and (getattr(op, "prune_module") is not None)

            n_merge += int(has_merge)
            n_prune += int(has_prune)

            if printed < max_print:
                print(f"[ATTACH] layer={li} block={bi} merge={has_merge} prune={has_prune}")
                printed += 1

        print(f"[ATTACH-SUM] total_blocks={total} merge_attached={n_merge} prune_attached={n_prune}")

    def debug_check_cached_delta(backbone, max_print=10):
        n_have = 0
        printed = 0
        for li, bi, blk in _iter_blocks(backbone):
            op = getattr(blk, "op", None)
            pm = getattr(op, "prune_module", None) if op is not None else None
            delta = getattr(pm, "cached_delta", None) if pm is not None else None

            if delta is not None:
                n_have += 1
                if printed < max_print:
                    print(f"[DELTA] layer={li} block={bi} cached_delta shape={tuple(delta.shape)} dtype={delta.dtype}")
                    printed += 1
        print(f"[DELTA-SUM] cached_delta_nonnull_blocks={n_have}")


    # 1) 백본 찾기
    backbone = _get_backbone(model)

    # 2) prune/merge 실제로 달렸는지 출력
    debug_print_attach(backbone, max_print=30)

    # 3) forward 1회 돌려서 cached_delta 생기는지 확인
    #    - train_loader에서 배치 하나 뽑아서 이미지 텐서만 사용
    device = next(model.parameters()).device
    batch = next(iter(train_loader))

    # ReID 로더는 보통 (img, pid, camid, viewid, ...) 형태라서 첫 번째를 이미지로 가정
    imgs = batch[0].to(device, non_blocking=True)

    model.eval()
    with torch.no_grad():
        _ = model(imgs)  # forward 1회

    debug_check_cached_delta(backbone, max_print=20)

    model.train()
  

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )
