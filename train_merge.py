import os
import argparse
import random
import numpy as np

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from config import get_config
from models import build_model
from models.TEvmamba import CrossEntropyLabelSmooth, TripletLoss
from utils.logger import create_logger
from timm.utils import AverageMeter


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ------------------------- Argparse -------------------------
parser = argparse.ArgumentParser(description='Train Person ReID with VMamba')
#parser.add_argument('--cfg', type=str, default='./configs/vssm/vmambav2v_tiny_224.yaml', help='config file')
#parser.add_argument('--cfg', type=str, default='./configs/vssm/vmambav2v_small_224.yaml', help='config file')
parser.add_argument('--cfg', type=str, default='./configs/vssm/vmambav2v_base_224.yaml', help='config file')
parser.add_argument('--opts', default=None, nargs='+', help="Modify config options like MODEL.NAME newname")
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# ------------------------- Fix Seed -------------------------
def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(args.seed)

# ------------------------- Config & Logger -------------------------
config = get_config(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(config.OUTPUT, exist_ok=True)

logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")
logger.info(f"Training config:\n{config}")


# ------------------------- Dataset -------------------------
transform = transforms.Compose([
    transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.4))
])

train_dataset = datasets.ImageFolder(config.DATA.TRAIN_PATH, transform)
train_loader = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=config.DATA.NUM_WORKERS)

num_classes = len(train_dataset.classes)
config.defrost()
config.MODEL.NUM_CLASSES = num_classes
config.MODEL.PRETRAINED = ""

config.freeze()
logger.info(f"Number of classes (person IDs): {num_classes}")

# ------------------------- Model -------------------------
model = build_model(config).to(device)

total_blocks = sum(len(layer.blocks) for layer in model.layers)
print(f" Total number of VSSBlocks in VMamba: {total_blocks}")

# ------------------------- Loss, Optimizer, Scaler -------------------------
criterion_list = [CrossEntropyLabelSmooth(), TripletLoss(margin=0.5)]
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.TRAIN.BASE_LR,
    weight_decay=config.TRAIN.WEIGHT_DECAY
)
scaler = GradScaler(enabled=config.AMP_ENABLE)

# ------------------------- Training Loop -------------------------
accumulation_steps = 4 # 가상 batch size = 8 x 4 = 32
start_time = time.time()
logger.info("Start training")

torch.autograd.set_detect_anomaly(True)


for epoch in range(config.TRAIN.EPOCHS):

    model.train()
    loss_meter = AverageMeter()
    correct, total = 0, 0

    optimizer.zero_grad()

    for step, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        with autocast(enabled=config.AMP_ENABLE):
            logits, features = model(images)
            features = nn.functional.normalize(features, p=2, dim=1)
            loss_ce = criterion_list[0](logits, targets)
            loss_tri = criterion_list[1](features, targets)
            loss = loss_ce + loss_tri
            loss = loss / accumulation_steps
        

        scaler.scale(loss).backward()


        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * accumulation_steps, targets.size(0))
        _, preds = torch.max(logits, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        if step % config.PRINT_FREQ == 0:
            logger.info(
                f"Epoch [{epoch}] Step [{step}/{len(train_loader)}] "
                f"Loss: {loss_meter.val:.4f} Avg: {loss_meter.avg:.4f}"
            )

    acc = correct / total
    logger.info(f"Epoch {epoch}: Loss: {loss_meter.avg:.4f}, Accuracy: {acc:.4f}")

# Record end time and print total duration
end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

# ------------------------- Save Model -------------------------
save_path = os.path.join(config.OUTPUT, f"{config.MODEL.NAME}.pth")
torch.save(model.state_dict(), save_path)
logger.info(f"Model saved at {save_path}")


