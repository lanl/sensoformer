# src/utils/positional_encodings.py
from __future__ import annotations
import math
import torch
import torch.nn as nn


class PositionalEncodings:
    """
    - Learned and AbsoluteSin return an additive tensor (1, L, d_model).
      Use: x = x + pe(x)

    - RoPE modifies attention Q/K. It exposes apply(q, k).
      q,k are expected shape: (B, H, L, head_dim)
    """

    class Learned(nn.Module):
        """Learned absolute positional embedding table."""
        def __init__(self, d_model: int, max_len: int = 4096, init_std: float = 0.02):
            super().__init__()
            self.max_len = max_len
            self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * init_std)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, L, d)
            L = x.size(1)
            if L > self.max_len:
                raise ValueError(f"Learned PE: L={L} > max_len={self.max_len}. Increase max_len.")
            return self.pos_emb[:L].unsqueeze(0).to(device=x.device, dtype=x.dtype)  # (1,L,d)

    class AbsoluteSin(nn.Module):
        """Fixed sinusoidal absolute positional encoding (non-learnable)."""
        def __init__(self, d_model: int, max_len: int = 4096, base: float = 10000.0):
            super().__init__()
            self.max_len = max_len

            pe = torch.zeros(max_len, d_model, dtype=torch.float32)
            pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (max_len,1)

            div = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(base) / d_model)
            )  # (d/2,)

            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)

            self.register_buffer("pe", pe, persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            L = x.size(1)
            if L > self.max_len:
                raise ValueError(f"AbsoluteSin PE: L={L} > max_len={self.max_len}. Increase max_len.")
            return self.pe[:L].unsqueeze(0).to(device=x.device, dtype=x.dtype)  # (1,L,d)

    class RoPE(nn.Module):
        """
        Rotary Position Embedding.

        Apply inside attention to q,k after projection:
          q,k: (B, H, L, head_dim)
        """
        def __init__(self, head_dim: int, max_len: int = 4096, base: float = 10000.0, rotary_dim: int | None = None):
            super().__init__()
            self.head_dim = head_dim
            self.max_len = max_len
            self.rotary_dim = rotary_dim if rotary_dim is not None else head_dim
            if self.rotary_dim % 2 != 0:
                raise ValueError("RoPE: rotary_dim must be even.")
            if self.rotary_dim > head_dim:
                raise ValueError("RoPE: rotary_dim cannot exceed head_dim.")

            inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

            # cached cos/sin for speed
            self.register_buffer("_cos", torch.empty(0), persistent=False)
            self.register_buffer("_sin", torch.empty(0), persistent=False)
            self._cached_len = 0

        def _build_cache(self, L: int, device, dtype):
            if L <= self._cached_len and self._cos.device == device and self._cos.dtype == dtype:
                return
            if L > self.max_len:
                raise ValueError(f"RoPE: L={L} > max_len={self.max_len}. Increase max_len.")

            t = torch.arange(L, device=device, dtype=self.inv_freq.dtype)  # (L,)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)             # (L, rotary_dim/2)
            self._cos = freqs.cos().to(dtype=dtype)                       # (L, rotary_dim/2)
            self._sin = freqs.sin().to(dtype=dtype)                       # (L, rotary_dim/2)
            self._cached_len = L

        @staticmethod
        def _rotate_half(x: torch.Tensor) -> torch.Tensor:
            # x: (..., rotary_dim)
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return torch.stack([-x2, x1], dim=-1).flatten(-2)

        def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # q,k: (B,H,L,head_dim)
            B, H, L, D = q.shape
            device, dtype = q.device, q.dtype
            self._build_cache(L, device, dtype)

            rd = self.rotary_dim
            q_rot, q_pass = q[..., :rd], q[..., rd:]
            k_rot, k_pass = k[..., :rd], k[..., rd:]

            cos = self._cos[:L].unsqueeze(0).unsqueeze(0)  # (1,1,L,rd/2)
            sin = self._sin[:L].unsqueeze(0).unsqueeze(0)

            # expand cos/sin to (1,1,L,rd) by interleaving
            cos = torch.stack([cos, cos], dim=-1).flatten(-2)
            sin = torch.stack([sin, sin], dim=-1).flatten(-2)

            q_rot = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
            k_rot = (k_rot * cos) + (self._rotate_half(k_rot) * sin)

            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)
            return q, k