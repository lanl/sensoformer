import random, math, torch

class MaskBank:
    def __init__(self, mask_ratios=[0.1, 0.5, 0.15, 0.15]):
        self.mask_ratios = mask_ratios

    def mask_pointwise(self, x):
        r = self.mask_ratios[0]
        return (torch.rand_like(x) < r), r

    def mask_timewise(self, x):
        # pick ONE row, then mask some columns in that row based on mask_ratio
        L, C = x.shape
        r = self.mask_ratios[1]

        t = random.randrange(L)  # one row
        k = int(round(r * C))
        k = max(1, min(k, C))
        if C > 1:
            k = min(k, C - 1)  # avoid masking the entire row when possible

        cols = random.sample(range(C), k)
        mask = torch.zeros(L, C, dtype=torch.bool, device=x.device)
        mask[t, cols] = True
        return mask, r

    def mask_channelwise(self, x):
        # pick ONE column, then mask some timesteps in that column based on mask_ratio
        L, C = x.shape
        r = self.mask_ratios[2]

        c = random.randrange(C)  # one column
        k = int(round(r * L))
        k = max(1, min(k, L))
        if L > 1:
            k = min(k, L - 1)  # avoid masking the entire column when possible

        rows = random.sample(range(L), k)
        mask = torch.zeros(L, C, dtype=torch.bool, device=x.device)
        mask[rows, c] = True
        return mask, r

    def mask_blockwise(self, x):
        # fixed AREA = 15% of total area, placed anywhere
        L, C = x.shape
        r = self.mask_ratios[3]

        total = L * C
        area = max(1, int(round(r * total)))
        if total > 1:
            area = min(area, total - 1)  # avoid full-mask when possible

        aspect = math.exp(random.uniform(math.log(0.3), math.log(3.0)))  # log-uniform
        h = int(round(math.sqrt(area * aspect)))
        w = int(round(math.sqrt(area / aspect)))
        h = max(1, min(h, L))
        w = max(1, min(w, C))

        t0 = random.randrange(L - h + 1)
        c0 = random.randrange(C - w + 1)

        mask = torch.zeros(L, C, dtype=torch.bool, device=x.device)
        mask[t0:t0+h, c0:c0+w] = True
        return mask, r

    def select_mask(self, x, mask_type):
        if mask_type == 'point':
            return self.mask_pointwise(x)
        elif mask_type == 'time':
            return self.mask_timewise(x)
        elif mask_type == 'channel':
            return self.mask_channelwise(x)
        elif mask_type == 'block':
            return self.mask_blockwise(x)
        else:
            raise ValueError('mask type not defined')