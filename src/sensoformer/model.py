import torch.nn as nn
import torch
from sensoformer.layers.transformer import EncoderBlock
from sensoformer.layers.set_encoder_decoder import SetEncoder, SetDecoder
from sensoformer.layers.positional_encoding import PositionalEncodings

class SensOFormer(nn.Module):
    def __init__(self, seq_len, dims, head, mlp_dim, depth, dropout,
                 slots_encoder=2, mslots_decoder=2, pool_type ='mean',
                 SENTINEL = 0.0, 
                 lora_r_attn=0, lora_r_mlp=0, lora_alpha=None, lora_p=0.0,
                 flash_attention=False,
                 pos_type = "learned",
                 pe_max_len = None):
        super().__init__()
        
        # projection
        # self.input_proj  = nn.Linear(num_features+1, dims)
        self.encoder = SetEncoder(d_model=dims, heads=head, slots=slots_encoder, 
                                  pool_type=pool_type,
                                  flash_attention=flash_attention)

        # Positional Encoding selection
        self.pos_type = pos_type.lower()
        max_len = pe_max_len or seq_len

        if self.pos_type == "learned":
            self.pos_enc = PositionalEncodings.Learned(d_model=dims, max_len=seq_len)
        elif self.pos_type == "absolute":
            self.pos_enc = PositionalEncodings.AbsoluteSin(d_model=dims, max_len=max_len)
        elif self.pos_type == "rope":
            self.pos_enc = PositionalEncodings.RoPE(head_dim=dims // head, max_len=max_len)
        else:
            raise ValueError(f"Unknown pos_type='{pos_type}'. Use: learned | absolute | rope | none")
        
        # Transformer backbone 
        self.transformer_blocks = nn.ModuleList([
                                  EncoderBlock(dims, head, mlp_dim, dropout,
                                               lora_r_attn = lora_r_attn, # PRETRAIN: keep 0 (dormant)
                                               lora_r_mlp = lora_r_mlp, # PRETRAIN: keep 0 (dormant)
                                               lora_alpha = lora_alpha, # PRETRAIN: keep None
                                               lora_p = lora_p,
                                               flash_attention=flash_attention)         
                                  for _ in range(depth)
                                  ])
        
        # projection
        # self.output_proj = nn.Linear(dims, num_features)
        self.decoder = SetDecoder(d_model=dims, heads=head, mem_slots=mslots_decoder,
                                 flash_attention=flash_attention)
        
        # distinguish mask
        self.SENTINEL = SENTINEL

        # # tag info
        # self.tag_dim = tag_dim  # can be 1 or more --> represent random tag added to each channel
        # self.tag_proj = nn.Linear(tag_dim, dims, bias=False)
        # self.tag_scale = tag_scale
        
    def _encode_context(self, x, mask):
        """Build the temporal latent and channel context used by the decoder."""
        x_fill = x.masked_fill(mask, self.SENTINEL)
        m_bit  = mask.to(x.dtype)

        feat = torch.stack([x_fill, m_bit], dim=-1)   # (B,L,C,2)
        
        # +++ encoder +++
        z_time, kv = self.encoder(feat)   # z_time: (B,L,d), kv: (B,L,C,d)

        # +++ per-channel context embedding e_c (masked-aware pooling) +++
        obs = (~mask)                                 # (B,L,C) bool: observed & real
        obs_f = obs.to(kv.dtype)                   # float mask for pooling

        den = obs_f.sum(dim=1).clamp_min(1.0)              # (B,C)
        e_c = (kv * obs_f.unsqueeze(-1)).sum(dim=1) / den.unsqueeze(-1)   # (B,C,d)

        # decoder queries: makes them channel-distinct even when kv[t,c] is identical
        q_tokens = kv + e_c.unsqueeze(1)                   # (B,L,C,d)

        # +++ transformer backbone +++
        # positional embedding
        if self.pos_type == "learned" or self.pos_type == "absolute":
            x_pos = self.pos_enc(z_time)  # (1,L,d)
            z = z_time + x_pos
            # transformer
            for blk in self.transformer_blocks:
                z = blk(z)                                                # [B,L,d]
        elif self.pos_type == "rope":
            z = z_time
            for blk in self.transformer_blocks:
                z = blk(z, pos_enc=self.pos_enc)
        else:
            raise ValueError(f"Unknown pos_type='{self.pos_type}'. Use: learned | absolute | rope")
        
        return z, kv, e_c, q_tokens

    def encode(self, x, mask):
        """Return the encoder/processor temporal latent with shape ``(B, L, dims)``."""
        z, _, _, _ = self._encode_context(x, mask)
        return z

    def forward(self, x, mask, return_z_only=False):
        """
        Args:
            x: Input tensor with shape ``(B, L, C)``.
            mask: Boolean corruption mask with shape ``(B, L, C)``.
            return_z_only: Preserve the legacy latent-only return tuple.
        """
        z, kv, e_c, q_tokens = self._encode_context(x, mask)

        if return_z_only:
            return None, z, None, None, None
        
        # +++ decode using q_tokens and z +++
        # decoder (linear projection)
        x_out = self.decoder(z, q_tokens)                                      # (B,L,C)
        #print(f'Decoder output:{x_out.shape}')
    
        return x_out, z, kv, e_c, q_tokens


# Backward-compatible class alias for code that imported the original name.
MaskedTransformer = SensOFormer
