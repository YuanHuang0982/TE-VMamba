# =========================================================
# AdaptivePairMergeGate.py  
# - even(a_k) <-> odd(b_k) 고정 페어만 고려 (O(P))
# - similarity: cosine (normalized dot)
# - theta: two-layer MLP + sigmoid
# - merge rule: sim > theta  
# - merge result: (a+b)/2
# - 구현은 GPU-friendly: prefix-sum + scatter/scatter_add (배치 loop 제거)
# =========================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


def do_nothing(x, mode=None):
    return x


def _norm(x: torch.Tensor, dim=-1, eps=1e-12):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


class FixedPairThresholdMerge(nn.Module):
    """
    논문 Eq.(6)-(7) 형태:
      s_k = cos(a_k, b_k)
      theta = sigmoid( MLP( mean(metric) ) )
      merge if s_k > theta
      z_k = (a_k + b_k)/2
    """

    def __init__(
        self,
        theta_min: float = 0.0,
        theta_max: float = 1.0,
        gate_hidden: int = 64,
        tau_init=0.1
    ):
        super().__init__()
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.gate_hidden = int(gate_hidden)

        # lazy init (C 모름)
        self.gate_head: Optional[nn.Module] = None
        # tau_init은 (0,1)이어야 함
        eps = 1e-6
        t = float(max(eps, min(1.0 - eps, tau_init)))
        # sigmoid(theta_base) = tau_init 되도록 logit으로 초기화
        self.theta_base = nn.Parameter(torch.tensor(math.log(t / (1.0 - t)), dtype=torch.float32))


        # logging
        self.last_theta: Optional[float] = None
        self.keep_ratio_est: Optional[float] = None

    def _theta_from_gate(self, metric: torch.Tensor) -> torch.Tensor:
        """
        metric: (B, T, C)
        theta: (B,) in [theta_min, theta_max]
        """
        B, T, C = metric.shape
        if self.gate_head is None:
            self.gate_head = nn.Sequential(
                nn.Linear(C, self.gate_hidden),
                nn.GELU(),
                nn.Linear(self.gate_hidden, 1),
            ).to(metric.device)

        g = metric.mean(dim=1)  # (B, C)
        raw = self.gate_head(g).squeeze(-1)  # (B,)
        theta = torch.sigmoid(self.theta_base + raw)  # (B,) in (0,1)
        theta = theta.clamp(min=self.theta_min, max=self.theta_max)
        return theta

    def forward(
        self,
        metric: torch.Tensor,            # (B, T, C)
        class_token: bool = False,
        distill_token: bool = False,
    ):
        B, T, C = metric.shape
        protected = (1 if class_token else 0) + (1 if distill_token else 0)

        # (T-protected) 홀수면 마지막 제거 (페어 맞춤)
        if (T - protected) % 2 == 1:
            metric = metric[:, :-1, :]
            T = metric.size(1)

        P = (T - protected) // 2
        if P <= 0:
            return do_nothing, {"merge_mask": None, "ratio": 1.0, "theta": 0.0}

        # 1) cosine similarity
        with torch.no_grad():
            m = _norm(metric)
            a = m[:, protected:, :][:, ::2, :]   # (B,P,C)
            b = m[:, protected:, :][:, 1::2, :]  # (B,P,C)
            sim = (a * b).sum(dim=-1)            # (B,P)

        # 2) theta via gate (two-layer MLP + sigmoid)
        theta = self._theta_from_gate(metric)            # (B,)
        mask_bool = (sim > theta.unsqueeze(-1))          # (B,P) True=merge

        # logging
        with torch.no_grad():
            self.last_theta = float(theta.mean().item())
            # keep ratio(토큰 수 기준): merge면 1개, no-merge면 2개
            exp_keep_per_pair = 1.0 * mask_bool + 2.0 * (~mask_bool)
            self.keep_ratio_est = float((exp_keep_per_pair.float().mean().item() / 2.0))

        # 3) order-preserving merge (벡터화)
        def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
            """
            x: (B, T, C)
            return: (B, T', C)  (batch마다 T' 다르면 max로 패딩)
            """
            Bx, Tx, Cx = x.shape
            if protected > 0:
                prot = x[:, :protected, :]
                x_ = x[:, protected:, :]
            else:
                prot = None
                x_ = x

            if (x_.shape[1] % 2) == 1:
                x_ = x_[:, :-1, :]

            src = x_[:, 0::2, :]   # (B,P,C)
            dst = x_[:, 1::2, :]   # (B,P,C)
            P_ = src.shape[1]

            mask = mask_bool.to(x_.device)
            if mask.shape[1] != P_:
                raise RuntimeError(f"mask P={mask.shape[1]} != src P={P_}")

            if mode == "mean":
                pair = (src + dst) * 0.5
            elif mode == "sum":
                pair = (src + dst)
            elif mode == "amax":
                pair = torch.maximum(src, dst)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            token1 = torch.where(mask.unsqueeze(-1), pair, src)  # (B,P,C)

            # out positions (prefix-sum)
            extra = (~mask).long()                      # unmerged면 1
            shift = torch.cumsum(extra, dim=1) - extra  # 이전 extra 개수
            base = torch.arange(P_, device=x_.device)[None, :]   # (1,P)
            pos1 = base + shift                         # (B,P)
            pos2 = pos1 + 1                             # (B,P) unmerged에서만 의미
            out_len = P_ + extra.sum(dim=1)             # (B,)
            out_max = int(out_len.max().item())

            out = x_.new_zeros(Bx, out_max, Cx)

            # (1) token1은 scatter_ 유지해도 안전 (pos1은 항상 0..out_len-1)
            out.scatter_(1, pos1.unsqueeze(-1).expand(-1, -1, Cx), token1)

            # (2) token2는 "unmerged만" 뽑아서 index_add (merge된 pos2는 아예 안 씀)
            unm = (~mask)  # (B,P) bool
            if unm.any():
                # out을 (B*out_max, C)로 펴서 1D index로 더하기
                out_flat = out.view(Bx * out_max, Cx)

                b = torch.arange(Bx, device=x_.device)[:, None].expand(Bx, P_)  # (B,P)
                idx2 = (b * out_max + pos2).masked_select(unm).long()           # (Nu,)
                val2 = dst.masked_select(unm.unsqueeze(-1)).view(-1, Cx)        # (Nu,C)

                # 안전 체크 (디버그용, 터지면 여기서 원인 바로 확정)
                # assert int(idx2.max().item()) < Bx * out_max and int(idx2.min().item()) >= 0

                out_flat.index_add_(0, idx2, val2)


            if prot is not None:
                out = torch.cat([prot, out], dim=1)
            return out

        return merge, {
            "merge_mask": mask_bool,
            "ratio": float(mask_bool.float().mean().item()),
            "theta": float(theta.mean().item()),
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
