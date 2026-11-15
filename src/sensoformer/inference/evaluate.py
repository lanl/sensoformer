import torch
from sensoformer.inference.data import split_sequence, stitch_splits
from sensoformer.data.augmentation import BatchAugmentor
from sensoformer.inference.metrics import ImputationMetrics

def test_model(
        model, 
        test_data_list,
        missing_data_list,
        seq_len,
        device,
        run_batches=None,
    ):
    
    model.eval()
    sample_visuals = []
    metrics = ImputationMetrics()

    masked_mse_total, unmasked_mse_total = 0.0, 0.0
    masked_mae_total, unmasked_mae_total = 0.0, 0.0
    masked_mae_all = []
    masked_mse_all = []

    masked_mre_total, unmasked_mre_total = 0.0, 0.0
    masked_rel_l2_total, unmasked_rel_l2_total = 0.0, 0.0
    masked_mre_all = []
    masked_rel_l2_all = []

    # Masked channel-wise range-normalized MAE percentage
    masked_nmae_range_pct_total = 0.0
    masked_nmae_range_pct_all = []

    # Masked MAPE percentage
    masked_mape_pct_total = 0.0
    masked_mape_pct_all = []

    masked_sse_total, unmasked_sse_total = 0.0, 0.0
    masked_count_total, unmasked_count_total = 0, 0

    x_mrwa_all = []  # for baseline imputation
    
    n_batches = 0

    with torch.no_grad():
        for batch_idx, test_data in enumerate(test_data_list[:run_batches]):
            
            orig_x = test_data
            orig_mask = missing_data_list[batch_idx].bool()
            #print(f'[Before Split] Batch {batch_idx}: original X shape: {orig_x.shape}, mask shape: {orig_mask.shape}')

            # split into un-overlapping windows
            splited_x, splited_mask = split_sequence(test_data, missing_data_list[batch_idx], seq_len=seq_len)

            # predict on each split
            y_pred_allsplits = []
            for x_split, m_split in zip(splited_x, splited_mask):

                # add batch dim
                x, mask = x_split.unsqueeze(0).to(device), m_split.unsqueeze(0).to(device) 

                # normalize on the fly
                x, min_val, max_val, ch_ok_full = BatchAugmentor.minmaxnorm_01_on_fly(x, mask)

                # pass through model
                y_pred, _, _, _, _ = model(x, mask)  

                # unnormalize on the fly
                y_pred = BatchAugmentor.minmaxunnorm_01_on_fly(y_pred, mask, min_val, max_val, ch_ok_full)

                # collect predictions for this split
                y_pred_allsplits.append(y_pred.squeeze(0).detach().cpu())  # remove batch dim

            # stitch back together
            stitched_sequence = stitch_splits(y_pred_allsplits)
            assert stitched_sequence.shape == orig_x.shape, \
            f"Stitched sequence shape {stitched_sequence.shape} does not match original shape {orig_x.shape}"

            # calculate mse and mae on masked and unmasked points
            sq = (stitched_sequence - orig_x) ** 2
            ae = (stitched_sequence - orig_x).abs()

            masked_idx = orig_mask
            unmasked_idx = ~orig_mask

            n_batches += 1

            # --- macro masked/unmasked MSE ---
            masked_mse = sq[masked_idx].mean() if masked_idx.any() else torch.tensor(0.0)
            unmasked_mse = sq[unmasked_idx].mean() if unmasked_idx.any() else torch.tensor(0.0)

            masked_mse_total += masked_mse
            unmasked_mse_total += unmasked_mse

            # --- macro masked/unmasked MAE ---
            masked_mae = ae[masked_idx].mean() if masked_idx.any() else torch.tensor(0.0)
            unmasked_mae = ae[unmasked_idx].mean() if unmasked_idx.any() else torch.tensor(0.0)

            masked_mae_total += masked_mae
            unmasked_mae_total += unmasked_mae

            # --- micro masked/unmasked MSE and MAE ---
            masked_sse_total += sq[masked_idx].sum()
            unmasked_sse_total += sq[unmasked_idx].sum()
            masked_count_total += masked_idx.sum().item()
            unmasked_count_total += unmasked_idx.sum().item()

            # --- normalized metrics ---
            masked_mre = metrics.mre(stitched_sequence, orig_x, masked_idx)
            unmasked_mre = metrics.mre(stitched_sequence, orig_x, unmasked_idx)

            masked_rel_l2 = metrics.relative_l2(stitched_sequence, orig_x, masked_idx)
            unmasked_rel_l2 = metrics.relative_l2(stitched_sequence, orig_x, unmasked_idx)

            # ---- masked channel-wise range NMAE percentage ---
            masked_nmae_range_pct = metrics.nmae_channel_range_pct(
                stitched_sequence, orig_x, masked_idx, time_dim=0,)

            # ---- masked MAPE percentage ---
            masked_mape_pct = metrics.mape_pct(
                stitched_sequence, orig_x,masked_idx,)

            masked_mre_total += masked_mre
            unmasked_mre_total += unmasked_mre

            masked_rel_l2_total += masked_rel_l2
            unmasked_rel_l2_total += unmasked_rel_l2

            # ---- masked channel-wise range NMAE percentage ---
            masked_nmae_range_pct_total += masked_nmae_range_pct

            # ---- masked MAPE percentage ---
            masked_mape_pct_total += masked_mape_pct

            # collect some samples for visualization 
            sample_visuals.append((orig_x.detach().cpu(),
                                   orig_mask.detach().cpu(),
                                   stitched_sequence.detach().cpu())) # across multiple batches
            
            # collect masked MSE and MAE and normalized metrics for all batches
            masked_mse_all.append(masked_mse.detach().cpu().item())
            masked_mae_all.append(masked_mae.detach().cpu().item())
            masked_mre_all.append(masked_mre.detach().cpu().item())
            masked_rel_l2_all.append(masked_rel_l2.detach().cpu().item())

            # ---- masked channel-wise range NMAE percentage ---
            masked_nmae_range_pct_all.append(masked_nmae_range_pct.detach().cpu().item())

            # ---- masked MAPE percentage ---
            masked_mape_pct_all.append(masked_mape_pct.detach().cpu().item())

            # baseline imputation calculation (replace masked with average of previous and next)
            x_mrwa, _ = metrics.baseline_calc(orig_x, orig_mask, time_dim=1)
            x_mrwa_all.append(x_mrwa.detach().cpu())

    avg_masked_mse, avg_unmasked_mse = masked_mse_total / n_batches, unmasked_mse_total / n_batches

    avg_masked_mae, avg_unmasked_mae = masked_mae_total / n_batches, unmasked_mae_total / n_batches

    avg_masked_mse_micro = masked_sse_total / max(masked_count_total, 1)
    avg_unmasked_mse_micro = unmasked_sse_total / max(unmasked_count_total, 1)

    avg_masked_mre = masked_mre_total / n_batches
    avg_unmasked_mre = unmasked_mre_total / n_batches

    avg_masked_rel_l2 = masked_rel_l2_total / n_batches
    avg_unmasked_rel_l2 = unmasked_rel_l2_total / n_batches

    # NEW
    avg_masked_nmae_range_pct = masked_nmae_range_pct_total / n_batches
    avg_masked_mape_pct = masked_mape_pct_total / n_batches

    return (
        sample_visuals,

        masked_mae_all,
        masked_mse_all,
        masked_mre_all,
        masked_rel_l2_all,
        masked_nmae_range_pct_all,
        masked_mape_pct_all,

        avg_masked_mae.detach().cpu().item(),
        avg_masked_mse.detach().cpu().item(),
        avg_masked_mre.detach().cpu().item(),
        avg_masked_rel_l2.detach().cpu().item(),
        avg_masked_nmae_range_pct.detach().cpu().item(),  
        avg_masked_mape_pct.detach().cpu().item(),
    )
