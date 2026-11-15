import torch.nn as nn
from sensoformer.layers.lora_linear import LoRALinear
from sensoformer.layers.sdpa import ScaledDotProductAttention
from sensoformer.layers.positional_encoding import PositionalEncodings

class LoRAMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, 
                 rank=0, alpha=None, p=0.0, flash_attention=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.dropout    = dropout

        # Projections; LoRA is dormant if rank=0
        self.q = LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha, p=p)
        self.k = LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha, p=p)
        self.v = LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha, p=p)
        self.o = LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha, p=p)

        # manual attention module (replaces SDPA)
        self.sdpa = ScaledDotProductAttention(dropout=dropout, 
                                flash_attention=flash_attention)
        
    def forward(self, *inputs, pos_enc = None):
        """
        Drop-in signatures:
          - y, _ = attn(x)                 # self-attention
          - y, _ = attn(q, k, v)           # explicit qkv
        Shapes: (B, L, C)
        """
        if len(inputs) == 1:
            q_in = k_in = v_in = inputs[0]
        elif len(inputs) == 3:
            q_in, k_in, v_in = inputs
        else:
            raise TypeError(f"LoRAMHA.forward expected 1 or 3 inputs, got {len(inputs)}")

        Bq, Lq, Cq = q_in.shape
        Bk, Lk, Ck = k_in.shape
        Bv, Lv, Cv = v_in.shape

        # sanity checks
        assert Bq == Bk == Bv, "Batch sizes of q,k,v must match"
        assert Cq == Ck == Cv == self.embed_dim, "embed_dim mismatch"

        h, d = self.num_heads, self.head_dim

        # project to heads
        q = self.q(q_in).contiguous().view(Bq, Lq, h, d).transpose(1, 2)   # (B, h, Lq, d)
        k = self.k(k_in).contiguous().view(Bk, Lk, h, d).transpose(1, 2)   # (B, h, Lk, d)
        v = self.v(v_in).contiguous().view(Bv, Lv, h, d).transpose(1, 2)   # (B, h, Lk, d)

        # apply RoPE if pos_enc is provided
        if isinstance(pos_enc, PositionalEncodings.RoPE) and (Lq == Lk):
                q, k = pos_enc.apply(q, k)

        # SDPA (masks optional)
        y, attn = self.sdpa(q, k, v)  # (B,h,Lq,d)
        # print(f'LoRAMHA attention output shape: {y.shape}')

        assert y.size(2) == q.size(2), f"SDPA must return Lq; got {y.shape}, q={q.shape}"

        # merge heads
        y = y.transpose(1, 2).contiguous().view(Bq, Lq, Cq)
        y = self.o(y)

        # match nn.MultiheadAttention return type: (output, attn_weights)
        return y, attn
