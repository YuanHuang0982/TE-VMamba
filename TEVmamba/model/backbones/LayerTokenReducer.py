# =========================================================
# LayerTokenReducer.py  
# - PRUNE: |Δ| 기반, theta = mu - alpha*sigma (alpha learnable)
# - 배치 loop 제거: prefix-sum + scatter_add
# - MERGE: AdaptivePairMergeGate.FixedPairThresholdMerge 사용
# =========================================================

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .AdaptivePairMergeGate import FixedPairThresholdMerge


# -------------------------------
# 공통 유틸
# -------------------------------
def _flatten_bchw_to_btc(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    B, C, H, W = x.shape
    tok = x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
    return tok, H, W


def _restore_tokens_to_grid(y_tok: torch.Tensor, H: int, W: int, prefer_axis: str = "W") -> torch.Tensor:
    B, K, C = y_tok.shape
    if K <= 0:
        return y_tok.new_zeros(B, C, H, W)

    if prefer_axis == "W":
        Hc, Wc = H, (K + H - 1) // H
    else:
        Wc, Hc = W, (K + W - 1) // W

    need = Hc * Wc - K
    if need > 0:
        y_tok = torch.cat([y_tok, y_tok.new_zeros(B, need, C)], dim=1)

    y = y_tok[:, :Hc * Wc, :].reshape(B, Hc, Wc, C).permute(0, 3, 1, 2).contiguous()
    return y


# =========================================================
# ① PRUNE MODULE (Δ 기반 adaptive pruning)
#   논문: theta = mu - alpha*sigma (alpha learnable)
# =========================================================
class LayerTokenPrune(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_layer: nn.Module,
        layer_idx: int = 1,
        alpha_init: float = 0.7,   # 초기값
        min_keep_ratio: float = 0.05,
        alpha_positive: bool = True,  # alpha>0 강제 여부(권장)
    ):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.layer_idx = layer_idx
        self.min_keep_ratio = float(min_keep_ratio)
        self.alpha_positive = bool(alpha_positive)

        # learnable alpha (논문 "learnable threshold parameter")
        self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_init)))

        # logging/cache
        self.cached_delta = None
        self.keep_ratio_est = None
        self.last_theta = None

    def _alpha(self):
        if self.alpha_positive:
            return F.softplus(self.alpha_raw)  # >0
        return self.alpha_raw

    def forward(
        self,
        x_bchw: torch.Tensor,
        delta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if delta is None:
            raise ValueError("LayerTokenPrune requires delta input")

        B, C, H, W = x_bchw.shape
        L = H * W
        tok = x_bchw.permute(0, 2, 3, 1).reshape(B, L, C)

        # delta -> delta_tok: (B, L)
        if delta.dim() == 4:
            if delta.shape[-2:] == (H, W):
                delta_tok = delta.abs().mean(dim=1).reshape(B, L)
            elif delta.shape[-1] == L:
                delta_tok = delta.abs().mean(dim=(1, 2))
            else:
                raise ValueError(f"Unexpected 4D delta shape {delta.shape} for H,W={H,W}, L={L}")
        elif delta.dim() == 3:
            if delta.shape[-1] == L:
                delta_tok = delta.abs().mean(dim=1)
            else:
                raise ValueError(f"Unexpected 3D delta shape {delta.shape}, expected last dim L={L}")
        else:
            raise ValueError(f"Unexpected delta shape: {delta.shape}")

        if delta_tok.shape[1] != L:
            raise ValueError(f"delta_tok length {delta_tok.shape[1]} != L {L}. delta shape={delta.shape}")

        # importance = |Δ|
        imp = delta_tok.abs()  # (B,L)

        # theta = mu - alpha*sigma
        mu = imp.mean(dim=1, keepdim=True)
        sigma = imp.std(dim=1, keepdim=True)
        alpha = self._alpha()
        theta = mu - alpha * sigma

        mask = imp >= theta  # (B,L) bool

        # 최소 유지 토큰 수(안정성)
        min_keep = max(1, int(L * self.min_keep_ratio))

        keep_cnt = mask.sum(dim=1)         # (B,)
        need_fix = keep_cnt < min_keep     # (B,)

        if need_fix.any():
            # topk로 min_keep 확보 (벡터화)
            topk_idx = imp.topk(k=min_keep, dim=1).indices  # (B,min_keep)
            topk_mask = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)
            topk_mask.scatter_(1, topk_idx, True)
            mask = torch.where(need_fix[:, None], topk_mask, mask)

        keep_cnt = mask.sum(dim=1)               # (B,)
        max_len = int(keep_cnt.max().item())     # python int

        # prefix-sum으로 압축 위치
        pos = mask.long().cumsum(dim=1) - 1      # (B,L)
        pos = pos.clamp_min(0)                  # keep 아닌 곳도 0

        # scatter_add로 compact (B, max_len, C)
        val = tok * mask.unsqueeze(-1).to(tok.dtype)       # keep 아니면 0
        idx = pos.unsqueeze(-1).expand(B, L, C).long()

        tok_new = tok.new_zeros(B, max_len, C)
        tok_new.scatter_add_(1, idx, val)

        # grid 복원 (논문상 토큰 수 줄었다는 효과를 유지하기 위해 "패딩 최소")
        K = tok_new.size(1)
        H_new = H
        W_new = math.ceil(K / H_new)

        need = H_new * W_new - K
        if need > 0:
            tok_new = torch.cat([tok_new, tok_new.new_zeros(B, need, C)], dim=1)

        y = tok_new.view(B, H_new, W_new, C).permute(0, 3, 1, 2).contiguous()

        if not torch.isfinite(y).all():
            y = torch.nan_to_num(y)

        self.keep_ratio_est = float((keep_cnt.float() / L).mean().item())
        self.last_theta = float(theta.mean().item())
        self.cached_delta = delta.detach()

        return y, {
            "keep_ratio": self.keep_ratio_est,
            "theta": self.last_theta,
            "alpha": float(alpha.detach().item()),
        }


# =========================================================
# ② MERGE MODULE (FixedPairThresholdMerge 기반)
# =========================================================
class LayerTokenMerge(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_layer: nn.Module,
        layer_idx: int = 1,
        merge_gate_hidden: int = 64,
        tau_init=0.1
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = dim
        self.norm = norm_layer(dim)

        self.merge_core = FixedPairThresholdMerge(
            gate_hidden=merge_gate_hidden,
            theta_min=0.0,
            theta_max=1.0,
            tau_init=tau_init,
        )
        self.last_merge_stats = None

    def _get_merge_fn(self, metric_tokens, hw, direction=None, **kwargs):
        B, T, C = metric_tokens.shape
        H, W = hw

        # pairing 가능한 영역만
        H_eff, W_eff = H - (H % 2), W - (W % 2)
        x2d = metric_tokens.view(B, H, W, C)[:, :H_eff, :W_eff, :]

        # 논문처럼 “고정 pairing”만. 방향 교대는 구현 선택(논문이 강제하진 않음)
        if direction is None:
            direction = "horizontal" if (self.layer_idx % 2 == 0) else "vertical"

        if direction in ("horizontal", "h"):
            a, b = x2d[:, :, 0::2, :], x2d[:, :, 1::2, :]
        else:
            a, b = x2d[:, 0::2, :, :], x2d[:, 1::2, :, :]

        a = a.reshape(B, -1, C)
        b = b.reshape(B, -1, C)
        inter = torch.stack([a, b], dim=2).reshape(B, -1, C)  # even/odd interleave

        merge_core_fn, stats = self.merge_core.forward(metric=inter, **kwargs)
        self.last_merge_stats = stats

        def wrapped(x):
            if isinstance(x, (tuple, list)):
                x = x[0]

            x2 = x.permute(0, 2, 3, 1)[:, :H_eff, :W_eff, :]

            if direction in ("horizontal", "h"):
                ax, bx = x2[:, :, 0::2, :], x2[:, :, 1::2, :]
            else:
                ax, bx = x2[:, 0::2, :, :], x2[:, 1::2, :, :]

            inter2 = torch.stack([ax.reshape(B, -1, C), bx.reshape(B, -1, C)], dim=2).reshape(B, -1, C)

            y_tok = merge_core_fn(inter2, mode="mean")  # 논문: (a+b)/2
            prefer = "W" if W >= H else "H"
            return _restore_tokens_to_grid(y_tok, H, W, prefer_axis=prefer)

        return wrapped

    def forward(self, x_bchw: torch.Tensor, **kwargs):
        if isinstance(x_bchw, (tuple, list)):
            x_bchw = x_bchw[0]

        tok2, H2, W2 = _flatten_bchw_to_btc(x_bchw)

        if tok2.shape[1] > 0:
            fn_merge = self._get_merge_fn(tok2, hw=(H2, W2), **kwargs)
            merged = fn_merge(x_bchw)
            return merged, self.last_merge_stats

        return x_bchw, None
