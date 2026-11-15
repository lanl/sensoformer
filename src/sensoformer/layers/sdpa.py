import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0, flash_attention=False):
        super().__init__()
        self.dropout = float(dropout)
        self.flash_attention = flash_attention

    def forward(self, q, k, v, attn_mask=None):
        """
        q: (B, h, Lq, d)
        k: (B, h, Lk, d)
        v: (B, h, Lk, d)
        returns:
          y:    (B, h, Lq, d)   # follows Lq
          attn: (B, h, Lq, Lk)
        """

        d = q.shape[-1]

        q = q.contiguous() 
        k = k.contiguous() 
        v = v.contiguous()
        
        if self.flash_attention and q.is_cuda:
            dropout_p = self.dropout if self.training else 0.0
            try:
                # force FlashAttention (fp16/bf16 only)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
                return y, None
            except Exception:
                # if flash not supported -> fall back to manual below
                pass
        
        # logits: (B,h,Lq,Lk)
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, 
                                            torch.finfo(scores.dtype).min)
            else:
                scores = scores + attn_mask.to(dtype=scores.dtype, 
                                            device=scores.device)

        scores = scores - scores.max(dim=-1, keepdim=True).values  # stable softmax
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout if self.training else 0.0, 
                        training=self.training) # (B,h,Lq,Lk)

        # out: (B,h,Lq,d)
        y = torch.matmul(attn, v)  # (B,h,Lq,d)

        return y, attn
