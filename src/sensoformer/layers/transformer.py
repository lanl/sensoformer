import torch.nn as nn
from sensoformer.layers.lora_mha import LoRAMHA
from sensoformer.layers.lora_linear import LoRALinear

class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, lora_r_attn: int = 0, 
                 lora_r_mlp: int = 0, lora_alpha: int = None, lora_p: float = 0.0,
                 flash_attention=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LoRAMHA(dim, heads, dropout=dropout, rank=lora_r_attn, 
                            alpha=lora_alpha, p=lora_p,
                            flash_attention=flash_attention)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(LoRALinear(dim, mlp_dim),
                                 nn.GELU(), 
                                 nn.Dropout(dropout),
                                 LoRALinear(mlp_dim, dim), 
                                 nn.Dropout(dropout))
        
    def forward(self, x, pos_enc=None):
        # Attention block
        residual = x
        #print('Shape before norm and attentions', x.shape)
        x = self.norm1(x)
        #print('Shape after layer norm', x.shape)
        attn,_ = self.attn(x,x,x, pos_enc=pos_enc)
        #print('Shape after axial attention', x_attn.shape)
        x = residual + attn
        #print('Shape after residual + x_attn', x.shape)

        # MLP block
        residual = x
        x = self.norm2(x)
        #print('Shape after second layer norm', x.shape)
        x = self.mlp(x)
        #print('Shape after MLP', x.shape)
        x = residual + x
        
        return x
