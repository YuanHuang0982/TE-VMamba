import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


from PIL import Image, ImageOps, ImageDraw
import numpy as np

def _add_border(img: Image.Image, color, width: int = 8):
    # PIL.ImageOps.expand는 테두리 추가
    return ImageOps.expand(img, border=width, fill=color)

def _save_retrieval_strip_with_borders(
    query_path,
    gallery_paths,
    gallery_is_true,   # list[bool], gallery_paths와 길이 동일
    save_path,
    thumb_size=(128, 256),
    border_w=8,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Query (파란색 테두리)
    q = Image.open(query_path).convert("RGB").resize(thumb_size)
    q = _add_border(q, color=(0, 102, 255), width=border_w)  # blue

    imgs = [q]

    # Gallery (True=초록, False=빨강)
    for gp, is_true in zip(gallery_paths, gallery_is_true):
        g = Image.open(gp).convert("RGB").resize(thumb_size)
        color = (0, 200, 0) if is_true else (220, 0, 0)       # green / red
        g = _add_border(g, color=color, width=border_w)
        imgs.append(g)

    # 가로로 붙이기
    w, h = thumb_size
    w2, h2 = w + 2 * border_w, h + 2 * border_w
    canvas = Image.new("RGB", (w2 * len(imgs), h2), color=(20, 20, 20))
    for i, im in enumerate(imgs):
        canvas.paste(im, (w2 * i, 0))

    canvas.save(save_path)

def _normalize_img_paths(paths, root_dir):
    """
    paths가
      - 절대경로: 그대로
      - 상대경로: root_dir 기준으로 보정
      - basename: root_dir 아래를 재귀 탐색해서 찾음(비용 줄이기 위해 캐시)
    """
    # root_dir 내 파일 인덱스(한번만)
    file_index = {}
    if root_dir and os.path.isdir(root_dir):
        # 너무 비싸면 dataset_dir만 넣어도 됨 (예: /.../data/Occluded_REID)
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    # basename -> fullpath (중복 basename이 있으면 첫번째만)
                    file_index.setdefault(fn, os.path.join(dirpath, fn))

    out = []
    for p in paths:
        if p is None:
            out.append(p)
            continue

        # 이미 존재하는 절대/상대 경로면 그대로
        if os.path.isabs(p) and os.path.exists(p):
            out.append(p)
            continue
        if os.path.exists(p):
            out.append(os.path.abspath(p))
            continue

        # root_dir 기준으로 붙여보기
        cand = os.path.join(root_dir, p)
        if os.path.exists(cand):
            out.append(cand)
            continue

        # basename만 왔을 경우: 인덱스에서 찾기
        bn = os.path.basename(p)
        if bn in file_index and os.path.exists(file_index[bn]):
            out.append(file_index[bn])
            continue

        # 못 찾으면 원본 유지(디버깅)
        out.append(p)

    return out



def _visualize_from_distmat(distmat, q_pids, g_pids, q_camids, g_camids, q_paths, g_paths,
                            out_dir, topk=10):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "correct"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "incorrect"), exist_ok=True)

    num_q = distmat.shape[0]
    for i in range(num_q):
        q_pid = int(q_pids[i])
        q_cam = int(q_camids[i])

        order = np.argsort(distmat[i])  # 작은 거리 = 더 유사

        # junk 제거: 같은 pid + 같은 cam
        junk = (g_pids == q_pid) & (g_camids == q_cam)
        order = order[~junk[order]]

        top_idx = order[:topk]
        top_gallery_paths = [g_paths[j] for j in top_idx]

        r1_correct = False
        if len(top_idx) > 0:
            r1_correct = (int(g_pids[top_idx[0]]) == q_pid)

        sub = "correct" if r1_correct else "incorrect"
        save_path = os.path.join(out_dir, sub, f"{i:05d}.jpg")

        gallery_is_true = [int(g_pids[j]) == q_pid for j in top_idx]

        _save_retrieval_strip_with_borders(
            query_path=q_paths[i],
            gallery_paths=top_gallery_paths,
            gallery_is_true=gallery_is_true,
            save_path=save_path,
            thumb_size=(128, 256),
            border_w=8,
        )
def _pack_eval_triplet(feat, pid, camids):
    if isinstance(feat, (tuple, list)):
        feat = feat[-1]
    cam_for_eval = camids.detach().cpu().view(-1).tolist()
    return feat, pid, cam_for_eval




def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update(_pack_eval_triplet(feat, vid, camids))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update(_pack_eval_triplet(feat, vid, camids))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    img_path_list = []
    pid_list = []      # ### ADD
    camid_list = []    # ### ADD
    feat_list = []     # ### ADD  (시각화용)

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):

        # ✅ 첫 배치만 출력
        if n_iter == 0:
            print("=== DEBUG: first batch meta ===")

            # pid / camid는 list/tuple일 수 있으니 그냥 앞 10개 출력
            print("pid[:10]    =", list(pid)[:10])
            print("camid[:10]  =", list(camid)[:10])

            # camids / target_view는 보통 torch tensor
            if torch.is_tensor(camids):
                print("camids[:10] =", camids[:10].detach().cpu().tolist())
            else:
                print("camids[:10] =", list(camids)[:10])

            if torch.is_tensor(target_view):
                print("view[:10]   =", target_view[:10].detach().cpu().tolist())
            else:
                print("view[:10]   =", list(target_view)[:10])

            # (선택) 값 범위도 같이
            # camids가 진짜 카메라 id인지 감 잡기 좋음
            if torch.is_tensor(camids):
                print("camids unique in batch =", sorted(set(camids.detach().cpu().tolist())))
            if torch.is_tensor(target_view):
                print("view unique in batch   =", sorted(set(target_view.detach().cpu().tolist())))
            print("================================")


        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)

            feat = model(img, cam_label=camids, view_label=target_view)

            # 기존 evaluator 업데이트(지표 계산)
            evaluator.update(_pack_eval_triplet(feat, pid, camids))

            # ### ADD: 시각화용으로 feature/pid/cam/path를 같은 순서로 누적
            feat_list.append(feat.detach().cpu())
            img_path_list.extend(list(imgpath))

            # pid, camid는 현재 collate에서 tuple/list로 들어옴
            pid_list.extend([int(x) for x in pid])
            camid_list.extend([int(x) for x in camid])

    # 기존 지표 출력
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    # =============================
    # ### ADD: 시각화 (query vs top-k gallery)
    # val_loader가 query+gallery 순서로 붙어서 들어오므로:
    #   [0:num_query) -> query, [num_query:] -> gallery
    # =============================
    if getattr(cfg.TEST, "VISUALIZE", False):
        out_dir = getattr(cfg.TEST, "VIS_DIR", os.path.join(cfg.OUTPUT_DIR, "vis"))
        topk = int(getattr(cfg.TEST, "VIS_TOPK", 10))

        feats = torch.cat(feat_list, dim=0)

        # evaluator와 동일하게 norm 옵션 반영 (feat_norm == 'yes' 인 경우)
        if str(cfg.TEST.FEAT_NORM).lower() in ["yes", "true", "1"]:
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)

        qf = feats[:num_query]
        gf = feats[num_query:]

        q_paths = img_path_list[:num_query]
        g_paths = img_path_list[num_query:]

        # === ADD: 경로 정규화 (basename 에러 방지) ===
        # OCC_ReID면 실제 이미지들이 이 경로 아래에 있음: ROOT_DIR/Occluded_REID/...
        root_dir = cfg.DATASETS.ROOT_DIR
        dataset_dir = os.path.join(root_dir, "Occluded_REID")  # 너 OCC_ReID.dataset_dir 기준
        # 더 안전하게: dataset_dir가 없으면 root_dir 전체를 탐색
        search_root = dataset_dir if os.path.isdir(dataset_dir) else root_dir

        q_paths = _normalize_img_paths(q_paths, search_root)
        g_paths = _normalize_img_paths(g_paths, search_root)


        q_pids = np.asarray(pid_list[:num_query], dtype=np.int32)
        g_pids = np.asarray(pid_list[num_query:], dtype=np.int32)
        q_camids = np.asarray(camid_list[:num_query], dtype=np.int32)
        g_camids = np.asarray(camid_list[num_query:], dtype=np.int32)

        # 거리행렬: (Q,G)
        # - normalize 했으면 cosine 기반이므로 dist = 1 - dot
        # - normalize 안 했으면 euclidean을 쓰는게 보통인데, 여기선 안전하게 euclidean로 분기
        if str(cfg.TEST.FEAT_NORM).lower() in ["yes", "true", "1"]:
            distmat = (1.0 - (qf @ gf.t())).numpy()
        else:
            # euclidean
            distmat = torch.cdist(qf, gf, p=2).numpy()

        _visualize_from_distmat(
            distmat=distmat,
            q_pids=q_pids, g_pids=g_pids,
            q_camids=q_camids, g_camids=g_camids,
            q_paths=q_paths, g_paths=g_paths,
            out_dir=out_dir,
            topk=topk
        )
        logger.info(f"[VIS] saved retrieval visualization to: {out_dir}")

    return cmc[0], cmc[4]
