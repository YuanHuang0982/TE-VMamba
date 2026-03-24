from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = 0

# backbone
_C.MODEL.NAME = "vssm"              # make_model에서 assert로 걸어둔 값과 맞추기

# pretrained
_C.MODEL.PRETRAIN_CHOICE = "none"   
_C.MODEL.PRETRAIN_PATH = ""         

# drop path 
_C.MODEL.DROP_PATH_RATE = 0.2

# VSSM settings
_C.MODEL.VSSM = CN()
_C.MODEL.VSSM.IN_CHANS = 3
_C.MODEL.VSSM.PATCH_SIZE = 4
_C.MODEL.VSSM.EMBED_DIM = 128
_C.MODEL.VSSM.DEPTHS = [2, 2, 20, 2]

_C.MODEL.VSSM.SSM_D_STATE = 1
_C.MODEL.VSSM.SSM_DT_RANK = "auto"
_C.MODEL.VSSM.SSM_RATIO = 1.0
_C.MODEL.VSSM.SSM_CONV = 3
_C.MODEL.VSSM.SSM_CONV_BIAS = False
_C.MODEL.VSSM.SSM_FORWARDTYPE = "v05_noz"

_C.MODEL.VSSM.MLP_RATIO = 4.0
_C.MODEL.VSSM.DOWNSAMPLE = "v3"     
_C.MODEL.VSSM.PATCHEMBED = "v2"     
_C.MODEL.VSSM.NORM_LAYER = "ln2d"

# ---- 원본 vmamba config에서 쓰는 옵션들 ----
_C.MODEL.VSSM.SSM_ACT_LAYER = "silu"
_C.MODEL.VSSM.SSM_DROP_RATE = 0.0
_C.MODEL.VSSM.SSM_INIT = "v0"

_C.MODEL.VSSM.MLP_ACT_LAYER = "gelu"
_C.MODEL.VSSM.MLP_DROP_RATE = 0.0

_C.MODEL.VSSM.PATCH_NORM = True
_C.MODEL.VSSM.POSEMBED = False
_C.MODEL.VSSM.GMLP = False

_C.MODEL.VSSM.USE_CHECKPOINT = True


# neck / loss heads
_C.MODEL.NECK = "bnneck"
_C.MODEL.IF_WITH_CENTER = "no"

_C.MODEL.ID_LOSS_TYPE = "softmax"
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.METRIC_LOSS_TYPE = "triplet"

_C.MODEL.DIST_TRAIN = False
_C.MODEL.NO_MARGIN = False
_C.MODEL.IF_LABELSMOOTH = "on"
_C.MODEL.COS_LAYER = False


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [224, 224]
_C.INPUT.SIZE_TEST  = [224, 224]
_C.INPUT.PROB = 0.5
_C.INPUT.RE_PROB = 0.5
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD  = [0.229, 0.224, 0.225]
_C.INPUT.PADDING = 10

_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.0
# timm auto_augment string 예: 'rand-m9-mstd0.5-inc1', 'original', 'v0', 'none'
_C.AUG.AUTO_AUGMENT = 'none'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.NAMES = "market1501"
_C.DATASETS.ROOT_DIR = "../data"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.SAMPLER = "softmax"
_C.DATALOADER.NUM_INSTANCE = 16

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()

_C.SOLVER.OPTIMIZER_NAME = "AdamW"

_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.MAX_EPOCHS = 100
_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.BIAS_LR_FACTOR = 1
_C.SOLVER.SEED = 1234
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.MARGIN = 0.3

_C.SOLVER.CENTER_LR = 0.5
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# weight decay: VMamba/ViT류는 보통 ResNet보다 크게 주는 편
# 일단 보수적으로 0.01부터 시작 추천 (0.0005는 너무 약할 가능성 큼)
_C.SOLVER.WEIGHT_DECAY = 0.01
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0   

_C.SOLVER.WARMUP_EPOCHS = 5

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.EVAL_PERIOD = 10
_C.SOLVER.IMS_PER_BATCH = 64

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.RE_RANKING = False
_C.TEST.WEIGHT = ""
_C.TEST.NECK_FEAT = "after"
_C.TEST.FEAT_NORM = "yes"
_C.TEST.DIST_MAT = "dist_mat.npy"
_C.TEST.EVAL = False

# --- Visualization (retrieval strip) ---
_C.TEST.VISUALIZE = False          # True면 시각화 저장
_C.TEST.VIS_DIR = ""               # 비우면 OUTPUT_DIR/vis 사용
_C.TEST.VIS_TOPK = 10              # query당 top-k gallery 저장


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = ""


# -----------------------------------------------------------------------------
# TOKEN ADAPTATION (Prune / Merge)
# -----------------------------------------------------------------------------
_C.MODEL.TOKEN_ADAPT = CN()
_C.MODEL.TOKEN_ADAPT.ENABLE = False

_C.MODEL.TOKEN_ADAPT.TOKEN_PRUNE = False
_C.MODEL.TOKEN_ADAPT.TOKEN_MERGE = False

# global block index 
_C.MODEL.TOKEN_ADAPT.PRUNE_LAYERS = list(range(5, 7))
_C.MODEL.TOKEN_ADAPT.MERGE_LAYERS = list(range(9, 12))
_C.MODEL.TOKEN_ADAPT.MERGE_START = 8   # merge를 최소 몇 번째 블록부터 허용할지

# prune 관련
_C.MODEL.TOKEN_ADAPT.PRUNE_THRESHOLD = 0.25   
_C.MODEL.TOKEN_ADAPT.PRUNE_SIM = "cos_delta"
_C.MODEL.TOKEN_ADAPT.PRUNE_IMP = "l2"

# merge 관련 
_C.MODEL.TOKEN_ADAPT.MERGE_TAU_GATE = 0.5
_C.MODEL.TOKEN_ADAPT.MERGE_GATE_HIDDEN = 64

_C.MODEL.TOKEN_ADAPT.FINAL_THRESHOLD = 0.1
