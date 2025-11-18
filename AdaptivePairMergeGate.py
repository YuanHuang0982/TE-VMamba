# AdaptivePairMergeGate.py
# - MERGE: Fixed-pair threshold 기반 머징 코어
#   - even(a_k) <-> odd(b_k) 고정 페어만 고려 (O(P))
#   - θ = softplus(base + δ)
#   - p = σ((sim - θ)/τ), mask(logits >= 0) 
#   - mask=True이면 a를 b에 더해 b만 유지, no-merge면 a,b 둘 다 유지


import math
from typing import Callable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def do_nothing(x, mode=None):
    return x

def do_nothing_with_state(x, state_in=None):
    return x, state_in

def _norm(x: torch.Tensor, dim=-1, eps=1e-12):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


# =========================
# ADAPTIVE MERGE CORE (θ = gate(metric))
# =========================
class FixedPairThresholdMerge(nn.Module):
    def __init__(
        self,
        tau_gate: float = 0.1,
        theta_min: float = 0.0,
        theta_max: float = 2.0,
        gate_hidden: int = 64,
    ):
        super().__init__()
        self.tau_gate = float(tau_gate)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.gate_hidden = gate_hidden

        # MLP gate head
        self.tau_head = None  # lazy init: C dimension unknown at init

        # 로깅 변수
        self.last_theta: Optional[float] = None
        self.merge_prob_mean: Optional[float] = None
        self.keep_ratio_est: Optional[float] = None

    # 입력 기반 adaptive θ
    def _theta_from_gate(self, metric: torch.Tensor):
        B, T, C = metric.shape
        if self.tau_head is None:
            self.tau_head = nn.Sequential(
                nn.Linear(C, self.gate_hidden),
                nn.GELU(),
                nn.Linear(self.gate_hidden, 1),
            ).to(metric.device)

        g = metric.mean(dim=1)  # (B, C): 각 배치 feature 평균
        t = torch.sigmoid(self.tau_head(g)).squeeze(-1)  # (B,)
        theta = self.theta_min + (self.theta_max - self.theta_min) * t
        return theta  # (B,)

    def forward(
        self,
        metric: torch.Tensor,
        class_token: bool = False,
        distill_token: bool = False,
    ):
        B, T, C = metric.shape
        protected = (1 if class_token else 0) + (1 if distill_token else 0)

        # 홀수 토큰 보정
        if (T - protected) % 2 == 1:
            metric = metric[:, :-1, :]
            T = metric.size(1)

        P = (T - protected) // 2
        if P <= 0:
            return do_nothing, {"merge_mask": None, "ratio": 1.0, "theta": 0.0}

        # -----------------------
        # 1. 유사도 계산
        # -----------------------
        with torch.no_grad():
            m = _norm(metric)
            a = m[..., protected:, :][..., ::2, :]
            b = m[..., protected:, :][..., 1::2, :]
            sim = (a * b).sum(dim=-1)  # (B, P)

        # -----------------------
        # 2. Adaptive threshold 계산
        # -----------------------
        theta = self._theta_from_gate(metric)  # (B,)
        theta_b = theta.unsqueeze(-1)
        logits = (sim - theta_b) / max(self.tau_gate, 1e-6)
        mask_bool = (logits >= 0)

        # -----------------------
        # 3. 통계 저장
        # -----------------------
        with torch.no_grad():
            self.last_theta = float(theta.mean().item())
            self.merge_prob_mean = float(torch.sigmoid(logits).mean().item())
            exp_keep_per_pair = 1.0 * mask_bool + 2.0 * (~mask_bool)
            self.keep_ratio_est = float((exp_keep_per_pair.float().mean().item() / 2.0))

        # -----------------------
        # 4. 병합 함수 정의
        # -----------------------
        def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
            if protected > 0:
                prot = x[:, :protected, :]
                x_ = x[:, protected:, :]
            else:
                prot = x.new_zeros(B, 0, C)
                x_ = x

            if (x_.shape[1]) % 2 == 1:
                x_ = x_[:, :-1, :]

            src = x_[..., ::2, :]
            dst = x_[..., 1::2, :]
            avg_tokens = (src + dst) / 2.0

            merged_list = []
            for b in range(B):
                keep_src = ~mask_bool[b]
                merged = torch.cat([
                    src[b][keep_src],            # 병합되지 않은 src 유지
                    avg_tokens[b][mask_bool[b]], # 병합된 dst만 유지
                    dst[b][~mask_bool[b]]        # 병합되지 않은 dst 유지
                ], dim=0)
                merged_list.append(merged)

            # ✅ 배치 정규화를 위한 패딩
            max_len = max(m.size(0) for m in merged_list)
            padded = []
            for m in merged_list:
                pad_len = max_len - m.size(0)
                if pad_len > 0:
                    pad = m.new_zeros(pad_len, m.size(1))
                    m = torch.cat([m, pad], dim=0)
                padded.append(m)

            merged = torch.stack([torch.cat([prot[b], padded[b]], dim=0) for b in range(B)], dim=0)
            return merged

        # -----------------------
        # 5. 함수와 통계 반환
        # -----------------------
        return merge, {
            "merge_mask": mask_bool,
            "ratio": mask_bool.float().mean().item(),
            "theta": float(theta.mean().item())
        }



# ===== helpers =====
def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor = None):
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size

def merge_source(merge: Callable, x: torch.Tensor, source: torch.Tensor = None):
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
    source = merge(source, mode="amax")
    return source
