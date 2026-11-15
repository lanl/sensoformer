import h5py
import numpy as np
import torch
from sensoformer.data.masking import MaskBank

def get_test_data(h5_path):
    test_data_list = []
    with h5py.File(h5_path, "r") as f:
        g = f["data"]
        keys = sorted(g.keys(), key=int)   # ["0","1","2",...]
        for i, k in enumerate(keys):
            arr = g[k][...]  # read dataset into numpy
            if np.isnan(arr).any() or np.isinf(arr).any():
                raise ValueError(f"Non-finite values in dataset key {k}: "
                    f"n_nan={np.isnan(arr).sum()}, n_inf={np.isinf(arr).sum()}")
            test_data_list.append(torch.from_numpy(arr).float())
    return test_data_list


def generate_missing_data(test_data_list, mask_ratio=0.15, mask_type='point', 
                          output_directory=None, plot_example=True):
    mask_bank = MaskBank(mask_ratios = [mask_ratio, mask_ratio, mask_ratio, mask_ratio]) 
    missing_data_list = []
    for test_data in test_data_list:
        mask, _ = mask_bank.select_mask(test_data, mask_type=mask_type) # create a mask of the same shape as test_data
        missing_data_list.append(mask)
    
    y = test_data_list[0][:, 0].detach().cpu().numpy()
    m = missing_data_list[0][:, 0].detach().cpu().numpy()

    if plot_example:
        import os
        import matplotlib.pyplot as plt

        ### plot the first sample first channels with masked points highlighted ###
        plt.figure(figsize=(24, 6))
        
        miss_idx = np.where(m == 1)[0]   # use (m==0) if 0 means missing
        print("num masked:", len(miss_idx))

        plt.plot(y, '.-', label="original")

        # If masked y are NaN, plot markers at y=0 just so you can SEE where they are
        if np.any(np.isnan(y[miss_idx])):
            plt.scatter(miss_idx, np.zeros_like(miss_idx), marker="o", color = "tab:red", label="masked (NaN)")
        else:
            plt.scatter(miss_idx, y[miss_idx], marker="o", color = "tab:red", label="masked")

        plt.title("Masked points")
        plt.legend()
        plt.tight_layout()

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            outpath = os.path.join(output_directory, "missing_data_example.png")
        else:
            outpath = "missing_data_example.png"

        plt.savefig(outpath, dpi=150)
        plt.close()
        print("saved:", outpath)

    return missing_data_list

def split_sequence(test_data, missing_data, seq_len=512):
    split_data_list = []
    split_mask_list = []

    n = test_data.shape[0]

    # If shorter (or equal) than seq_len, return as-is
    if n <= seq_len:
        return [test_data], [missing_data]

    # Full splits
    n_splits = n // seq_len
    for s in range(n_splits):
        start = s * seq_len
        end = (s + 1) * seq_len
        split_data_list.append(test_data[start:end])
        split_mask_list.append(missing_data[start:end])

    # Leftover split (no padding)
    leftover_start = n_splits * seq_len
    if leftover_start < n:
        split_data_list.append(test_data[leftover_start:])
        split_mask_list.append(missing_data[leftover_start:])

    return split_data_list, split_mask_list

def stitch_splits(all_splits):
    stitched_sequence = torch.cat(all_splits, dim=0)
    return stitched_sequence
