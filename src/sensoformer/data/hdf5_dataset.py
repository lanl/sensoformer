import random
import torch
import h5py
from torch.utils.data import Dataset
from sensoformer.data.masking import MaskBank

class SequenceDataset(Dataset):
    # Each DataLoader worker is its own process.
    # This dict lives per-process, so each worker opens the file once and reuses it.
    _FILES = {}  # {h5_path: h5py.File}

    def __init__(self, data, seq_len=128, stride=64,
                 mask_ratios=[0.15,0.15,0.15,0.15]):

        # data is always (h5_path, key) where key selects one sequence in f["data"][key]
        self.h5_path, self.h5_key = map(str, data)
        self.seq_len, self.stride = seq_len, stride

        # masking setup
        self.mask_types = ['point','time','channel','block']
        self.mask_bank = MaskBank(mask_ratios=mask_ratios)

        # Read T once so we know how many sliding windows exist for this key.
        # (Open + close immediately here; safe in the main process.)
        with h5py.File(self.h5_path, "r") as f:
            T = f["data"][self.h5_key].shape[0]

        # Number of windows for this sequence
        self.n = max(1, (T - seq_len) // stride + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Convert window index -> start time
        s = idx * self.stride

        # Lazily open the H5 file in THIS worker process (if not already open)
        f = self._FILES.get(self.h5_path)
        if f is None:
            f = h5py.File(self.h5_path, "r")
            self._FILES[self.h5_path] = f

        # Grab one [L, C] window from this sequence key and convert to torch
        x = torch.as_tensor(f["data"][self.h5_key][s:s+self.seq_len], dtype=torch.float32)

        # Build a random mask and return
        mask_type = random.choice(self.mask_types)
        mask, r = self.mask_bank.select_mask(x, mask_type=mask_type)
        return x, mask, r, mask_type
