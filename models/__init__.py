import os
from functools import partial
import torch
from .TEvmamba import VSSM

def load_state_dict_flexible(model, state_dict):
    model_dict = model.state_dict()
    compatible_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    print(f"✅ Loaded {len(compatible_state_dict)} / {len(model_dict)} parameters from checkpoint.")
    model_dict.update(compatible_state_dict)
    model.load_state_dict(model_dict, strict=False)


def build_vssm_model(config, **kwargs):
    model_type = config.MODEL.TYPE
    if model_type in ["vssm"]:
        # ✅ backend 안전 처리
        backend = config.MODEL.VSSM.BACKEND
        if backend is not None:
            backend = backend.strip().lower()
        assert backend in [None, "oflex", "mamba", "torch"], f"Invalid backend: {backend}"

        print(f"\u2705 Using backend: {backend}")

        model = VSSM(
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            backend=backend,  # 안전하게 처리된 값 사용
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            # ===================
            posembed=config.MODEL.VSSM.POSEMBED,
            imgsize=config.DATA.IMG_SIZE,
            
            visualize=config.MODEL.VSSM.VISUALIZE,
            # =========================
            token_merge=config.MODEL.TOKEN_MERGE,        # ✅ 병합 기능을 켤지 여부 (기본 True)
            token_prune=config.MODEL.TOKEN_PRUNE,
             
            merge_layers=config.MODEL.MERGE_LAYERS,
            prune_layers=config.MODEL.PRUNE_LAYERS,
        )
        return model

    return None

def build_model(config, is_pretrain=False):
    model = None
    if model is None:
        model = build_vssm_model(config)
    if model is None:
        from .simvmamba import simple_build
        model = simple_build(config.MODEL.TYPE)
        
    # ✅ pretrained checkpoint 로딩 추가
    if config.MODEL.PRETRAINED:
        print(f"🔄 Loading pretrained weights from: {config.MODEL.PRETRAINED}")
        state_dict = torch.load(config.MODEL.PRETRAINED, map_location="cpu")
        
        # 🔧 'model' 키가 있으면 안쪽 딕셔너리로 들어가기
        if "model" in state_dict:
            state_dict = state_dict["model"]
        load_state_dict_flexible(model, state_dict)  # <- 중요 포인트!

    print(f"Selected backend: [{config.MODEL.VSSM.BACKEND}]")
    return model
