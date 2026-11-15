import torch

class ImputationMetrics:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def baseline_calc(self, x, mask, time_dim=1):
        """
        Replace masked points with avg(nearest unmasked left, nearest unmasked right) along time.
        Returns:
        x_in: imputed input
        avg_replace_mask: True where replacement happened (has both left & right neighbors)
        """
        x_in = x.clone()
        mask = mask.bool()

        # Broadcast mask to x shape if needed (e.g. [B,T] -> [B,T,1] -> [B,T,C])
        if mask.dim() < x.dim():
            while mask.dim() < x.dim():
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(x)

        T = x.size(time_dim)
        idx = torch.arange(T, device=x.device)

        # reshape idx to broadcast like x along time_dim
        view_shape = [1] * x.dim()
        view_shape[time_dim] = T
        idx = idx.view(*view_shape).expand_as(x)

        unmasked = (~mask)

        # last unmasked index to the left (forward cummax)
        left_idx = torch.where(unmasked, idx, torch.full_like(idx, -1))
        left_idx = left_idx.cummax(dim=time_dim).values

        # next unmasked index to the right (reverse + cummin + reverse)
        right_idx = torch.where(unmasked, idx, torch.full_like(idx, T))
        right_idx = right_idx.flip(time_dim).cummin(dim=time_dim).values.flip(time_dim)

        left_ok  = left_idx >= 0
        right_ok = right_idx < T

        avg_replace_mask = mask & left_ok & right_ok

        # gather left/right values
        left_val  = x.gather(time_dim, left_idx.clamp_min(0).long())
        right_val = x.gather(time_dim, right_idx.clamp_max(T-1).long())

        x_in[avg_replace_mask] = 0.5 * (left_val[avg_replace_mask] + right_val[avg_replace_mask])
        return x_in, avg_replace_mask

    def mre(self, pred, target, mask):
        """
        Mean Relative Error used by SAITS.

        Formula:
            MRE = sum(|pred - target| over mask) / sum(|target| over mask)
        """
        if not mask.any():
            return torch.tensor(0.0, device=pred.device)

        error_abs = (pred - target).abs()
        target_abs = target.abs()

        return error_abs[mask].sum() / (target_abs[mask].sum() + self.eps)

    def relative_l2(self, pred, target, mask):
        """
        Relative L2 error norm.

        Formula:
            Relative L2 = sqrt(sum((pred - target)^2 over mask))
                          / sqrt(sum(target^2 over mask))
        """
        if not mask.any():
            return torch.tensor(0.0, device=pred.device)

        error_sq = (pred - target) ** 2
        target_sq = target ** 2

        return torch.sqrt(error_sq[mask].sum()) / (
            torch.sqrt(target_sq[mask].sum()) + self.eps
        )
    
    def nmae_channel_range_pct(self, pred, target, mask, time_dim=0):
        """
        Masked channel-wise range-normalized MAE, returned as percentage.

        Assumes shape [T, C].

        Formula per channel c:
            NMAE_c% = 100 * mean(|pred - target| on masked points in channel c)
                            / (max(target[:, c]) - min(target[:, c]) + eps)

        Final:
            average NMAE_c% over valid channels.
        """
        if not mask.any():
            return torch.tensor(0.0, device=pred.device)

        error_abs = (pred - target).abs()
        mask_float = mask.to(error_abs.dtype)

        channel_min = target.min(dim=time_dim).values
        channel_max = target.max(dim=time_dim).values
        channel_range = channel_max - channel_min

        masked_error_sum = (error_abs * mask_float).sum(dim=time_dim)
        masked_count = mask_float.sum(dim=time_dim)

        valid_channel = (masked_count > 0) & (channel_range > self.eps)

        if not valid_channel.any():
            return torch.tensor(0.0, device=pred.device)

        channel_mae = masked_error_sum[valid_channel] / masked_count[valid_channel]
        channel_nmae = channel_mae / (channel_range[valid_channel] + self.eps)

        return 100.0 * channel_nmae.mean()
    
    def mape_pct(self, pred, target, mask, eps=1e-8):
        """
        Masked Mean Absolute Percentage Error, returned as percentage.

        Formula:
            MAPE% = 100 * mean(|pred - target| / (|target| + eps))
            computed only on masked points.
        """
        if not mask.any():
            return torch.tensor(0.0, device=pred.device)

        error_abs = (pred - target).abs()
        denom = target.abs().clamp_min(eps)

        return 100.0 * (error_abs[mask] / denom[mask]).mean()