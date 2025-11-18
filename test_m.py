import os
import argparse
import random
import numpy as np
import faiss
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import time

from utils import *
from metrics import *

from config import get_config
from models import build_model

import models.TEvmamba as VMD


# ------------------------- Argument Parsing & Setup -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Test Person ReID Model with VMamba on Market1501')
    parser.add_argument('--seed', default=42)
    parser.add_argument('--model_path', type=str, default="./output_M/Test_11/vssm1_base_0229s/baseline/vssm1_base_0229s.pth")
    parser.add_argument('--test_data', type=str, default="./dataset/market_1501")
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save_preds', type=str, default="./output_p/bbbb")
    return parser.parse_args()

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# ------------------------- Custom Dataset for Market1501 -------------------------
class Market1501Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = [f for f in os.listdir(root) if f.endswith('.jpg')]
        self.imgs.sort()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        fname = self.imgs[idx]
        path = os.path.join(self.root, fname)
        img = Image.open(path).convert('RGB')

        # 파일명: 0001_c1s1_001051_00.jpg
        pid = int(fname.split('_')[0])       # person ID
        camid = int(fname.split('_')[1][1])  # camera ID: c1 → 1

        if self.transform:
            img = self.transform(img)

        return img, pid, camid, fname

'''

# ------------------------- Custom Dataset for Occ -------------------------
class occ(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # 모든 하위 디렉토리 탐색
        self.imgs = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.endswith(".png"):
                    self.imgs.append(os.path.join(dirpath, f))

        self.imgs.sort()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = self.imgs[idx]
        fname = os.path.basename(path)
        img = Image.open(path).convert("RGB")

        # 파일명 예시: 0001_c1s1_001051_00.jpg
        pid = int(fname.split('_')[0])
        camid = int(fname.split('_')[1][1])

        if self.transform:
            img = self.transform(img)

        return img, pid, camid, fname
'''

# ------------------------- Feature Extraction -------------------------
def extract_feature(model, dataloader, device):
    imgs = torch.FloatTensor()
    features = torch.FloatTensor()

    #latencies = []  
    #peak_mems = [] 

    for data in tqdm(dataloader):
        img, pid, camid, fname = data
        img_copy = img.clone()
        imgs = torch.cat((imgs, img_copy), 0)
        img = img.to(device)

        # # ===================== PEAK MEMORY RESET =====================
        # torch.cuda.reset_peak_memory_stats()
        # torch.cuda.synchronize()

        # # ===================== latency 측정 시작 =====================
        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # torch.cuda.synchronize()
        # starter.record()

        with torch.no_grad():
            output = model(img)

        # ender.record()
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender)  # ms 단위
        # latencies.append(curr_time)
        # # ===================== latency 측정 끝 =========================
        # # ===================== peak memory 측정 ======================
        # peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB 단위
        # peak_mems.append(peak)
        # # =============================================================



        if isinstance(output, tuple):
            output = output[1]
        output = F.normalize(output, p=2, dim=1)
        features = torch.cat((features, output.cpu()), 0)

    # avg_latency = np.mean(latencies) / img.size(0)  # batch당 시간 → 이미지당 시간
    # avg_peak_mem = np.mean(peak_mems)                # ★ 평균 peak memory
    # print(f"### Avg latency per image: {avg_latency:.4f} ms")
    # print(f"### Avg peak memory per batch: {avg_peak_mem:.2f} MB")

    return features, imgs #, avg_latency, avg_peak_mem

def get_id(dataset):
    camera_id = []
    labels = []
    for _, pid, camid, _ in dataset:
        labels.append(pid)
        camera_id.append(camid)
    return camera_id, labels

def search(query, k=10):
    encoded_query = query.unsqueeze(dim=0).numpy()
    top_k = index.search(encoded_query, k)
    gallery_imgs_idxs = top_k[1][0].copy()
    top_k[1][0] = np.take(gallery_label, indices=top_k[1][0])
    return top_k, gallery_imgs_idxs

# ------------------------- Visualization -------------------------
def visualize(query_img, gallery_imgs, gallery_idxs, label, gallery_labels, save_path):
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    t = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize(size=(128, 48))
    ])
    plt.figure(figsize=(16., 6.))
    plt.subplot(1, 11, 1)
    img_tensor = query_img.clone()
    for i in range(3):
        img_tensor[i] = (img_tensor[i] * std[i]) + mean[i]
    x = t(img_tensor)
    x = np.array(x)
    plt.xticks([])
    plt.yticks([])
    plt.title("Query")
    plt.imshow(x)

    for j in range(10):
        img_tensor = gallery_imgs[gallery_idxs[j]].clone()
        for i in range(3):
            img_tensor[i] = (img_tensor[i] * std[i]) + mean[i]
        x = t(img_tensor)
        x = np.array(x)
        plt.subplot(1, 11, j + 2)
        plt.title("True" if gallery_labels[j] == label else "False")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# ------------------------- Main Execution -------------------------
args = parse_args()
fix_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
model_path = args.model_path
data_dir = args.test_data

# Record test start time
start_time = time.time()

# Data Transform
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets and Dataloaders
query_dataset = Market1501Dataset(os.path.join(data_dir, 'query'), transform)
gallery_dataset = Market1501Dataset(os.path.join(data_dir, 'bounding_box_test'), transform)

query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False)
'''

query_dataset = occ(os.path.join(data_dir, 'query'), transform)
gallery_dataset = occ(os.path.join(data_dir, 'gallery'), transform)

query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False)
'''
# Load model
class Args:
    cfg = './configs/vssm/vmambav2v_base_224.yaml'
    #cfg = './configs/vssm/vmambav2v_small_224.yaml'
    #cfg = './configs/vssm/vmambav2v_tiny_224.yaml'
    opts = None

args_cfg = Args()
config = get_config(args_cfg)
model = build_model(config).to(device)

ckpt = torch.load(model_path, map_location=device)
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)
model.eval()


# Extract features
query_feature, query_imgs = extract_feature(model, query_loader, device)
gallery_feature, gallery_imgs = extract_feature(model, gallery_loader, device)
# query_feature, query_imgs, query_latency, query_peak = extract_feature(model, query_loader, device)
# gallery_feature, gallery_imgs, gallery_latency, gallery_peak = extract_feature(model, gallery_loader, device)

'''
# ---------------- Query ----------------
VMD.ENABLE_TOKEN_ADAPTATION = False   # ✅ Query는 Token Adaptation 적용
query_feature, query_imgs, query_latency = extract_feature(model, query_loader, device)

# ---------------- Gallery ----------------
VMD.ENABLE_TOKEN_ADAPTATION = False  # ✅ Gallery는 원본 유지
gallery_feature, gallery_imgs, gallery_latency = extract_feature(model, gallery_loader, device)
'''

print(f"Query Latency per image: {query_latency:.4f} ms")
print(f"Gallery Latency per image: {gallery_latency:.4f} ms")
print(f"Query Peak Memory: {query_peak:.2f} MB")
print(f"Gallery Peak Memory: {gallery_peak:.2f} MB")

avg_total_latency = (query_latency + gallery_latency) / 2
avg_total_peak = (query_peak + gallery_peak) / 2

print(f"Overall Avg Latency per image: {avg_total_latency:.4f} ms")
print(f"Overall Avg Peak Memory: {avg_total_peak:.2f} MB")





# Get labels
gallery_cam, gallery_label = get_id(gallery_dataset)
query_cam, query_label = get_id(query_dataset)
gallery_label = np.array(gallery_label)
query_label = np.array(query_label)

# Build FAISS index
feature_dim = query_feature.shape[1]
index = faiss.IndexFlatIP(feature_dim)
index.add(gallery_feature.numpy())

# Make directories for visualizations
if args.visualize:
    os.makedirs(args.save_preds, exist_ok=True)
    os.makedirs(os.path.join(args.save_preds, "correct"), exist_ok=True)
    os.makedirs(os.path.join(args.save_preds, "incorrect"), exist_ok=True)

# Evaluation
rank1_score = 0
rank5_score = 0
ap_score = 0
count = 0

for query, label in zip(query_feature, query_label):
    query_img = query_imgs[count]
    output, gallery_idxs = search(query, k=10)

    r1 = rank1(label, output)
    rank1_score += r1
    rank5_score += rank5(label, output)
    ap_score += calc_map(label, output)

    #args.visualize = True

    if args.visualize:
        if r1:
            save_path = os.path.join(args.save_preds, "correct", f"{count:03d}.png")
            visualize(query_img, gallery_imgs, gallery_idxs, label, output[1][0], save_path)
        else:
            save_path = os.path.join(args.save_preds, "incorrect", f"{count:03d}.png")
            visualize(query_img, gallery_imgs, gallery_idxs, label, output[1][0], save_path)

    count += 1

# Final Results
print("Correct: {}, Total: {}, Incorrect: {}".format(rank1_score, count, count - rank1_score))
print("Rank1: %.3f, Rank5: %.3f, mAP: %.3f" % (
    rank1_score / count,
    rank5_score / count,
    ap_score / count
))

# Record end time and print test duration
end_time = time.time()
total_time = end_time - start_time
m, s = divmod(total_time, 60)
h, m = divmod(m, 60)
print(f"Total test time: {int(h)}h {int(m)}m {int(s)}s")