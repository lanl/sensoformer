import torch

class BatchAugmentor:
    @staticmethod
    # normalization on the fly (per batch)
    def mustdnorm_on_fly(x, mask, eps=1e-5):

        # x: (B,L,C), mask: (B,L,C) bool where True=masked(target)
        x32 = x.float()
        valid = (~mask).to(x32.dtype)                  # 1 where visible
        cnt = valid.sum(dim=1, keepdim=True)           # (B,1,C)
        cnt_safe = cnt.clamp(min=1.0)

        mu  = (x32 * valid).sum(dim=1, keepdim=True) / cnt_safe
        var = ((x32 - mu) ** 2 * valid).sum(dim=1, keepdim=True) / cnt_safe
        std = (var + eps).sqrt()

        # channels with zero visible points in this window
        ch_ok = (cnt > 0)                              # (B,1,C)
        mu  = mu.masked_fill(~ch_ok, 0.0)
        std = std.masked_fill(~ch_ok, 1.0)

        x_norm = (x32 - mu) / std
        ch_ok_full = ch_ok.expand_as(mask)             # (B,L,C)
        return x_norm, mu, std, ch_ok_full
    
    @staticmethod
    def minmaxnorm_01_on_fly(x, mask, eps=1e-5):
        x32 = x.float()
        valid = ~mask
        valid_f = valid.to(x32.dtype)

        cnt = valid_f.sum(dim=1, keepdim=True)   # (B,1,C)
        ch_ok = (cnt > 0)

        min_val = x32.masked_fill(~valid, float("inf")).amin(dim=1, keepdim=True)
        max_val = x32.masked_fill(~valid, float("-inf")).amax(dim=1, keepdim=True)

        range_val = (max_val - min_val).clamp(min=eps)
        range_val = range_val.masked_fill(~ch_ok, 1.0)

        x_norm = (x32 - min_val) / range_val
        x_norm = x_norm.clamp(0.0, 1.0)

        ch_ok_full = ch_ok.expand_as(mask)
        return x_norm, min_val, max_val, ch_ok_full
    
    @staticmethod
    def minmaxunnorm_01_on_fly(x_norm, mask, min_val, max_val, ch_ok_full, eps=1e-5):
        x32 = x_norm.float()

        # ch_ok is constant across time 
        ch_ok = ch_ok_full[:, :1, :]

        # Rebuild safe range/min for inverse transform
        range_val = (max_val - min_val).clamp(min=eps)
        range_val = range_val.masked_fill(~ch_ok, 1.0)

        min_val_safe = min_val.masked_fill(~ch_ok, 0.0)

        x_unnorm = x32 * range_val + min_val_safe

        return x_unnorm.to(dtype=x_norm.dtype)
    
    @staticmethod
    def minmaxnorm_11_on_fly(x, mask, eps=1e-5):
        # x: (B,L,C), mask: (B,L,C) bool where True=masked(target)
        x32 = x.float()
        valid = ~mask
        valid_f = valid.to(x32.dtype)

        cnt = valid_f.sum(dim=1, keepdim=True)   # (B,1,C)
        ch_ok = (cnt > 0)

        min_val = x32.masked_fill(~valid, float("inf")).amin(dim=1, keepdim=True)
        max_val = x32.masked_fill(~valid, float("-inf")).amax(dim=1, keepdim=True)

        range_val = (max_val - min_val).clamp(min=eps)
        range_val = range_val.masked_fill(~ch_ok, 1.0)

        x_01 = (x32 - min_val) / range_val
        x_m11 = x_01 * 2.0 - 1.0
        x_m11 = x_m11.clamp(-1.0, 1.0)

        ch_ok_full = ch_ok.expand_as(mask)
        return x_m11, min_val, max_val, ch_ok_full
        
    @staticmethod
    # add noise to the inputs 
    def add_noise(x, mask):
        
        B, L, C = x.shape
        device, dtype = x.device, x.dtype
        
        # randomly picking one noise level
        levels = [0.000, 0.005, 0.01, 0.015] 
        idx = torch.randint(len(levels), (), device=device).item()
        sigma = torch.tensor(levels[idx], device=device, dtype=dtype).view(1, 1, 1)

        # only add noise to UNMASKED positions
        noise_mask = ~mask                        # (B,L,C) bool

        noise = torch.randn_like(x) * sigma
        x_noisy = x + noise * noise_mask.to(dtype)
        
        return x_noisy, sigma.item()