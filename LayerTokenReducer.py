
# LayerTokenReducer 머징프루닝 나눈거
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from AdaptivePairMergeGate import FixedPairThresholdMerge
import math

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
# =========================================================
class LayerTokenPrune(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_layer: nn.Module,
        layer_idx: int = 1,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.layer_idx = layer_idx
        self.alpha = alpha
        self.cached_delta = None
        self.keep_ratio_est = None
        self.last_theta = None

    def forward(
        self,
        x_bchw: torch.Tensor,
        delta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if delta is None:
            raise ValueError("CrossLayerPrune requires delta input")

        B, C, H, W = x_bchw.shape
        L = H * W
        tok = x_bchw.permute(0, 2, 3, 1).reshape(B, L, C)

        # Δ → 중요도 계산 (정규화 안정화 포함)
        if delta.dim() == 4:
            delta_tok = delta.view(B, -1)
        elif delta.dim() == 3:
            delta_tok = delta.mean(dim=-1)
        else:
            raise ValueError(f"Unexpected delta shape: {delta.shape}")

        imp = delta_tok.abs()
        range_val = (imp.max(dim=1, keepdim=True)[0] - imp.min(dim=1, keepdim=True)[0]).clamp_min(1e-3)
        imp = (imp - imp.min(dim=1, keepdim=True)[0]) / range_val

        mu, sigma = imp.mean(1, keepdim=True), imp.std(1, keepdim=True)
        theta = mu - self.alpha * sigma
        keep_mask = imp >= theta

        pruned_tok, keep_ratios = [], []
        for b in range(B):
            #keep_idx = keep_mask[b].nonzero(as_tuple=False).squeeze(1)
            keep_idx = torch.where(keep_mask[b])[0]
            if keep_idx.numel() == 0:
                # 최소 5%는 유지 (안정성 보장)
                min_keep = max(1, int(L * 0.05))
                topk = imp[b].topk(min_keep, dim=0).indices
                keep_idx = torch.unique(topk)[:L]
            t_pruned = tok[b, keep_idx, :]
            pruned_tok.append(t_pruned)
            keep_ratios.append(t_pruned.size(0) / L)

        max_len = max(t.size(0) for t in pruned_tok)
        padded = []
        for t in pruned_tok:
            pad_len = max_len - t.size(0)
            if pad_len > 0:
                pad = t.new_zeros(pad_len, t.size(1))
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        tok_new = torch.stack(padded, dim=0)

        # 안전한 (H, W) 재계산
        K = tok_new.size(1)
        H_new = int(math.sqrt(K))
        W_new = math.ceil(K / H_new)

        # 필요 시 패딩
        need = H_new * W_new - K
        if need > 0:
            pad = tok_new.new_zeros(B, need, C)
            tok_new = torch.cat([tok_new, pad], dim=1)

        y = tok_new.view(B, H_new, W_new, C).permute(0, 3, 1, 2)

        # NaN 방지
        if not torch.isfinite(y).all():
            print(f"[WARN] Non-finite values detected at layer {self.layer_idx}")
            y = torch.nan_to_num(y)

        self.keep_ratio_est = float(sum(keep_ratios) / len(keep_ratios))
        self.last_theta = theta.mean().item()
        self.cached_delta = delta.detach()

        return y, {"keep_ratio": self.keep_ratio_est, "theta": self.last_theta}


# =========================================================
# ② MERGE MODULE (FixedPairThresholdMerge 기반)
# =========================================================
class LayerTokenMerge(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_layer: nn.Module,
        layer_idx: int = 1,
        merge_tau_gate: float = 0.1,
        merge_gate_hidden: int = 64,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = dim
        self.norm = norm_layer(dim)

        # learnable merge core
        self.merge_core = FixedPairThresholdMerge(
            tau_gate=merge_tau_gate,
            gate_hidden=merge_gate_hidden,
        )

    def _get_merge_fn(self, metric_tokens, hw, direction=None, **kwargs):
        B, T, C = metric_tokens.shape
        H, W = hw
        H_eff, W_eff = H - (H % 2), W - (W % 2)
        x2d = metric_tokens.view(B, H, W, C)[:, :H_eff, :W_eff, :]
        if direction is None:
            direction = "horizontal" if (self.layer_idx % 2 == 0) else "vertical"
        if direction in ("horizontal", "h"):
            a, b = x2d[:, :, 0::2, :], x2d[:, :, 1::2, :]
        else:
            a, b = x2d[:, 0::2, :, :], x2d[:, 1::2, :, :]
        a = a.reshape(B, -1, C)
        b = b.reshape(B, -1, C)
        inter = torch.stack([a, b], dim=2).reshape(B, -1, C)
        merge_core_fn, stats = self.merge_core.forward(metric=inter, **kwargs)
        self.last_merge_stats = stats  # 로그나 시각화용으로 보관

        def wrapped(x):
            if isinstance(x, (tuple, list)):
                x = x[0]
            x2 = x.permute(0, 2, 3, 1)[:, :H_eff, :W_eff, :]
            if direction in ("horizontal", "h"):
                ax, bx = x2[:, :, 0::2, :], x2[:, :, 1::2, :]
            else:
                ax, bx = x2[:, 0::2, :, :], x2[:, 1::2, :, :]
            inter = torch.stack([ax.reshape(B, -1, C), bx.reshape(B, -1, C)], dim=2).reshape(B, -1, C)


            
            y_tok = merge_core_fn(inter)
            prefer = "W" if W >= H else "H"
            return _restore_tokens_to_grid(y_tok, H, W, prefer_axis=prefer)
        return wrapped

    def forward(self, x_bchw: torch.Tensor, **kwargs):
        if isinstance(x_bchw, (tuple, list)):
            x_bchw = x_bchw[0]
        out = x_bchw

        # flatten
        tok2, H2, W2 = _flatten_bchw_to_btc(out)

        if tok2.shape[1] > 0:
            fn_merge = self._get_merge_fn(tok2, hw=(H2, W2), **kwargs)
            merged = fn_merge(out)
            B, C, Hm, Wm = merged.shape

            # ✅ FLOPs 진짜 줄이려면 여기서 그대로 반환
            return merged, self.last_merge_stats

        return out, None

