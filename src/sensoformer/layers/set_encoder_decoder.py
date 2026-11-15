import torch
import torch.nn as nn
from sensoformer.layers.lora_mha import LoRAMHA

# ------------------------------------------------------------
# Permutation-INVARIANT set encoder (channels-as-set per time)
# ------------------------------------------------------------

class SetEncoder(nn.Module):
    """
    (B, L, C) -> (B, L, d_model)
    Invariant to any permutation of the C channels.
    """
    def __init__(self, d_model: int, heads: int = 8, 
                 slots: int = 2, pool_type ='mean',
                 dropout=0.0, lora_r_attn=0, lora_alpha=None, lora_p=0.0,
                 flash_attention=False):
        super().__init__()
        self.d_model, self.slots, self.pool_type = d_model, slots, pool_type

        # 2d feature per channel per time-step. scalars: [mask_value, if-masked] -> token
        self.token = nn.Linear(2, d_model, bias=False)

        # Small learnable latent array that "reads" the set at each time step
        self.latent = nn.Parameter(torch.randn(1, slots, d_model))

        # Cross-attention: latent queries read channel tokens
        self.attn = LoRAMHA(embed_dim=d_model, num_heads=heads,
                            dropout=dropout, rank=lora_r_attn, 
                            alpha=lora_alpha, p=lora_p,
                            flash_attention=flash_attention)
        
        # gated pooling
        self.slot_score = nn.Linear(d_model, 1) if (pool_type == 'gated' and slots > 1) else None

        # Layer norm and output projection
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, L, C, 2) where last dim = [value_filled, mask_bit]
        """
        B, L, C, F = x.shape
        assert F == 2, f"Expected last dim=2, got {F}"

        x = x.reshape(B * L, C, F)                                # (B*L, C, 2)
        #print(f'Encoder input reshaped: {x.shape}')

        # Shared channel token embed: scalar -> token
        kv = self.token(x)                                        # (B*L, C, d_model)
        #print(f'Channel tokens: {kv.shape}')

        # Queries: latent per time step
        q  = self.latent.expand(B * L, self.slots, self.d_model)  # (B*L, S, d_model)
        #print(f'Latent queries: {q.shape}')

        # Cross-attention: latent queries read channel tokens
        y, _ = self.attn(q, kv, kv)                               # (B*L, S, d_model)
        #print(f'Attention output before pooling: {y.shape}')

        # ---- pool slots ----
        if self.slots == 1:
            y = y[:, 0]
        elif self.pool_type == 'mean':
            y = y.mean(dim=1)
        elif self.pool_type == 'gated':
            w = torch.softmax(self.slot_score(y).squeeze(-1), dim=1)  # (B*L, S)
            y = (y * w.unsqueeze(-1)).sum(dim=1)                      # (B*L, d)
        else:
            raise ValueError(f"Unknown pool_type='{self.pool_type}' (use 'mean' or 'gated')")

        y = y.view(B, L, self.d_model)
        y = self.norm(y)
        q_tokens = kv.view(B, L, C, self.d_model)
        return y, q_tokens

# ------------------------------------------------------------
# Permutation-EQUIVARIANT set decoder (outputs per channel)
# ------------------------------------------------------------

class SetDecoder(nn.Module):
    """
    Inputs:
      z         : (B, L, d_model)          time latents
      q_tokens  : (B, L, C, d_model)       per-channel queries (from encoder tokens)
    Output:
      y         : (B, L, C)
    """
    def __init__(self, d_model, heads=8, mem_slots=2,
                 dropout=0.0, lora_r_attn=0, lora_alpha=None, lora_p=0.0,
                 flash_attention=False):
        super().__init__()
        self.d_model   = d_model
        self.mem_slots = mem_slots or d_model  

        # Build a small memory from the latent at each (B,t)
        self.to_mem    = nn.Linear(d_model, self.mem_slots * d_model)

        # Cross-attention: per-channel queries read from memory
        self.attn = LoRAMHA(embed_dim=d_model, num_heads=heads, 
                            dropout=dropout, rank=lora_r_attn, 
                            alpha=lora_alpha, p=lora_p, 
                            flash_attention=flash_attention)
        
        # Layer norm and output projection
        self.norm      = nn.LayerNorm(d_model)

        # Project to scalar per channel
        self.to_scalar = nn.Linear(d_model, 1)

    def forward(self, z, q_tokens):
        '''
        memory slots controls how much “per-timestep information capacity” 
        you give the model to reconstruct all channels at that time step
        '''
        B, L, d_model = z.shape
        B2, L2, C, d_model_2 = q_tokens.shape
        assert (B2, L2, d_model_2) == (B, L, d_model), "q_tokens must be (B, L, C, d_model)"

        # Keys/values: memory slots from z_t
        mem = self.to_mem(z).view(B * L, self.mem_slots, d_model)   # (B*L, M, d_model)

        # Queries: per-channel tokens
        q = q_tokens.view(B * L, C, d_model)                        # (B*L, C, d_model)

        # Cross-attention per time step and channel
        out, _ = self.attn(q, mem, mem)                       # (B*L, C, d_model)
        out = self.norm(out)

        # Project to scalar per channel
        out = self.to_scalar(out).squeeze(-1)                 # (B*L, C)
        return out.view(B, L, C)                              # (B, L, C)
