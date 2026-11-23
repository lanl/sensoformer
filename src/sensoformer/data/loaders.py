import torch
import numpy as np
import random
import h5py
from sensoformer.data.hdf5_dataset import SequenceDataset
from torch.utils.data import DataLoader, ConcatDataset
from sensoformer.training.distributed import is_main_process
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

def _is_distributed():
    return dist.is_available() and dist.is_initialized()

def _seed_worker(worker_id):
    # make dataloader workers deterministic but different
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def DataLoaders(h5_paths, 
    seq_len=128, stride=4, 
    mask_ratios=[0.15,0.15,0.15,0.15],
    batch_size=32, keep_data = 1.0, 
    num_workers=1, pin_memory=True, seed=42):
    
    # ---- gather (path, key, T) across ALL files ----
    seq_refs = []
    for p in sorted(h5_paths):
        with h5py.File(p, "r") as f:
            g = f["data"]
            try:
                keys = sorted(g.keys(), key=int)
            except ValueError:
                keys = sorted(g.keys())

            for k in keys:
                seq_refs.append((p, k))

    # One SequenceDataset per original sequence (lazily reads windows from H5)
    parts = [SequenceDataset(ref, seq_len, stride, mask_ratios) for ref in seq_refs]

    # Split by key (sequence-level split) to avoid leakage between splits
    n_total = len(parts)

    # print some parts info for debugging
    # if is_main_process():
    #     print(f"Total sequences found across files: {n_total}")
    #     print("Example sequence refs:")
    #     for i in range(min(5, n_total)):
    #         with h5py.File(parts[i].h5_path, "r") as f:
    #             full_shape = f["data"][parts[i].h5_key].shape
    #         print(f"  {i}: {parts[i].h5_path} key {parts[i].h5_key} full shape {full_shape}")

    if n_total == 0:
        raise RuntimeError("No sequences found across supplied h5_paths")
    if n_total < 2:
        raise RuntimeError("Need at least 2 sequences for train/val split")

    n_val = max(1, int(round(n_total * 0.1)))
    n_val = min(n_val, n_total - 1)
    n_train = n_total - n_val

    g_torch = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g_torch).tolist()
    idx_train = perm[:n_train]
    idx_val   = perm[n_train:n_train+n_val]

    # keep only a fraction of training data
    keep_data = max(0.0, min(1.0, float(keep_data)))
    if keep_data < 1.0:
        if keep_data == 0.0:
            raise ValueError("keep_data must be > 0.0")
        n_keep = max(1, int(len(idx_train) * keep_data))
        idx_train = idx_train[:n_keep]

    # Flatten each split into one big dataset of windows using ConcatDataset
    train_ds = ConcatDataset([parts[i] for i in idx_train])
    val_ds = ConcatDataset([parts[i] for i in idx_val])

    # guardrails for empty datasets after seq_len filtering
    if len(train_ds) == 0:
        raise RuntimeError(f"Train dataset has 0 windows. "
            f"Check seq_len={seq_len} against your sequence lengths.")
    if len(val_ds) == 0:
        raise RuntimeError(f"Val dataset has 0 windows. "
            f"Check seq_len={seq_len} against your sequence lengths.")

    # +++ DDP sharding +++
    # DistributedSampler splits the GLOBAL index space across ranks.
    # Each rank sees a different subset of window indices.
    ddp = _is_distributed()
    if ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False) if len(train_ds) > 0 else None
        val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False) if len(val_ds) > 0 else None
        shuffle_train = False
        gen_for_loader = None
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True
        gen_for_loader = g_torch # for shuffling in DataLoader

    # +++ DataLoader workers +++
    # Workers parallelize __getitem__ calls *within each rank*.
    # Each worker opens its own H5 file handle lazily (safe).
    loader_train = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        worker_init_fn=_seed_worker, generator=gen_for_loader,
        persistent_workers=(num_workers > 0)
    )
    loader_val = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker, drop_last=False,
        persistent_workers=(num_workers > 0)
    )

    if is_main_process():
        print(f'Train parts:{len(idx_train)}, Val parts:{len(idx_val)}')
        print(f'Batches — Train:{len(loader_train)}, Val:{len(loader_val)}')

    return loader_train, loader_val
