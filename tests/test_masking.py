import random

import torch

from sensoformer.data.masking import MaskBank


def test_all_mask_types():
    random.seed(3)
    torch.manual_seed(3)
    x = torch.zeros(16, 8)
    bank = MaskBank(mask_ratios=[0.25, 0.5, 0.25, 0.2])

    for mask_type in ("point", "time", "channel", "block"):
        mask, ratio = bank.select_mask(x, mask_type)
        assert mask.shape == x.shape
        assert mask.dtype == torch.bool
        assert mask.any()
        assert not mask.all()
        assert 0.0 < ratio < 1.0
