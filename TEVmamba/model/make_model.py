import torch
import torch.nn as nn

import copy
from .backbones.vmamba import VSSM

from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

from .backbones import vmamba as vmamba_mod


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if getattr(m, "affine", False):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class VMambaReID(nn.Module):
    """
    TransReID 스타일 인터페이스를 유지한 VMamba(VSSM) ReID 모델.
    - train: return (cls_score, global_feat)
    - eval : return feat or global_feat (cfg.TEST.NECK_FEAT에 따라)
    """
    def __init__(self, num_classes, cfg):
        super().__init__()

        # ---- TransReID 쪽 설정 재사용 ----
        self.neck = cfg.MODEL.NECK                 # 'no' or 'bnneck'
        self.neck_feat = cfg.TEST.NECK_FEAT        # 'after' or 'before'
        self.cos_layer = cfg.MODEL.COS_LAYER       # bool
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        # ---- VMamba/VSSM 설정 ----
        vcfg = cfg.MODEL.VSSM
        ta = cfg.MODEL.TOKEN_ADAPT

        vmamba_mod.ENABLE_TOKEN_ADAPTATION = bool(ta.ENABLE)

    
        self.base = VSSM(
            patch_size=getattr(vcfg, "PATCH_SIZE", 4),
            in_chans=getattr(vcfg, "IN_CHANS", 3),
            num_classes=0,  # ReID에서는 보통 backbone head 사용 안 함
            depths=vcfg.DEPTHS,
            dims=[vcfg.EMBED_DIM, vcfg.EMBED_DIM*2, vcfg.EMBED_DIM*4, vcfg.EMBED_DIM*8]
                 if not hasattr(vcfg, "DIMS") else vcfg.DIMS,
            ssm_d_state=vcfg.SSM_D_STATE,
            ssm_dt_rank=vcfg.SSM_DT_RANK,
            ssm_ratio=vcfg.SSM_RATIO,
            ssm_conv=vcfg.SSM_CONV,
            ssm_conv_bias=vcfg.SSM_CONV_BIAS,
            forward_type=vcfg.SSM_FORWARDTYPE,
            mlp_ratio=vcfg.MLP_RATIO,
            drop_path_rate=cfg.MODEL.DROP_PATH_RATE,
            norm_layer=vcfg.NORM_LAYER,
            downsample_version=vcfg.DOWNSAMPLE,      
            patchembed_version=vcfg.PATCHEMBED,      
            imgsize=cfg.INPUT.SIZE_TRAIN[0] if isinstance(cfg.INPUT.SIZE_TRAIN, (list, tuple)) else cfg.INPUT.SIZE_TRAIN,

            # 여기부터 TOKEN_ADAPT 매핑
            token_merge=bool(ta.ENABLE and ta.TOKEN_MERGE),
            token_prune=bool(ta.ENABLE and ta.TOKEN_PRUNE),
            merge_layers=list(ta.MERGE_LAYERS),
            prune_layers=list(ta.PRUNE_LAYERS),
            merge_start=int(ta.MERGE_START),
            prune_threshold=float(ta.PRUNE_THRESHOLD),
            prune_sim=str(ta.PRUNE_SIM),
            prune_imp=str(ta.PRUNE_IMP),
            final_threshold=float(ta.FINAL_THRESHOLD),
        )


        self.channel_first = self.base.channel_first
        self.in_planes = self.base.num_features
        self.num_classes = num_classes

        # ---- BNNeck ----
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # ---- pretrained 로딩 ----
        if cfg.MODEL.PRETRAIN_CHOICE in ("imagenet", "pretrained") and cfg.MODEL.PRETRAIN_PATH:
            self._load_backbone_pretrained(cfg.MODEL.PRETRAIN_PATH)


        # ---- Classifier / Metric head ----
        if self.ID_LOSS_TYPE == 'arcface':
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

    def forward_features(self, x):
        x = self.base.patch_embed(x)
        if self.base.pos_embed is not None:
            pos = self.base.pos_embed
            pos = pos.permute(0, 2, 3, 1) if (not self.base.channel_first) else pos
            x = x + pos
        for layer in self.base.layers:
            x = layer(x)
        return x


    def pool(self, feat_map):
        if feat_map.dim() != 4:
            raise RuntimeError(f"Expected 4D feat map, got {feat_map.shape}")
        if not self.channel_first:
            feat_map = feat_map.permute(0, 3, 1, 2).contiguous()
        global_feat = nn.functional.adaptive_avg_pool2d(feat_map, 1).flatten(1)  # (B,C)
        return global_feat

    def forward(self, x, label=None, cam_label=None, view_label=None):
        feat_map = self.forward_features(x)
        global_feat = self.pool(feat_map)

        feat = self.bottleneck(global_feat) if (self.neck == 'bnneck') else global_feat

        if self.training:
            cls_score = self.classifier(feat, label) if self.ID_LOSS_TYPE in ('arcface','cosface','amsoftmax','circle') else self.classifier(feat)
            return cls_score, global_feat
        else:
            return feat if (self.neck_feat == 'after') else global_feat

    def load_param(self, trained_path: str):
        ckpt = torch.load(trained_path, map_location="cpu")

        # 자주 있는 포맷들 흡수
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            elif "model" in ckpt:
                ckpt = ckpt["model"]

        # DDP로 저장된 module. prefix 제거
        new_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_ckpt[k] = v

        incompatible = self.load_state_dict(new_ckpt, strict=False)
        print("Load ckpt:", trained_path)
        print("Missing keys:", incompatible.missing_keys)
        print("Unexpected keys:", incompatible.unexpected_keys)
    
    def _load_backbone_pretrained(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            elif "model" in ckpt:
                ckpt = ckpt["model"]

        # DDP prefix 제거
        new = {}
        for k, v in ckpt.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new[k] = v

        # ===== 분류 head 제거 (shape mismatch 방지) =====
        drop_keys = [
            "classifier.head.weight",
            "classifier.head.bias",
            "head.weight",
            "head.bias",
        ]
        for k in drop_keys:
            if k in new:
                new.pop(k)

        for k in list(new.keys()):
            if k.startswith("classifier.head.") or k.startswith("head."):
                new.pop(k)
        # ===============================================

        incompatible = self.base.load_state_dict(new, strict=False)
        print("Load backbone ckpt:", path)
        print("Missing keys:", incompatible.missing_keys)
        print("Unexpected keys:", incompatible.unexpected_keys)


def make_model(cfg, num_class, camera_num, view_num):
    model = VMambaReID(num_class, cfg)
    print('===========building VMamba(VSSM) ReID===========')
    return model