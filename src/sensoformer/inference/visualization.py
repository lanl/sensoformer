import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# Visualization of some test samples
def visualize_test_samples(
        sample_visuals, 
        output_directory,               
        test_mask_ratio = 0.05, 
        viz_batches=1, 
        viz_unnorm=False, 
        viz_channels=10, 
        viz_split = 1024, 
        model_name="XTi", 
        dataset_id=0):

    # randomly select some batches to visualize 
    viz_test_batches = np.random.choice(len(sample_visuals), size=viz_batches, replace=False)
    print(f"Visualizing {len(viz_test_batches)} random batches from test set: {viz_test_batches}")

    for bidx in viz_test_batches:
        org_x, masks, outputs = sample_visuals[bidx] # batch index
        print(f"sample shape: {org_x.shape}, mask shape: {masks.shape}, output shape: {outputs.shape}")

        # if plot on normalized scale
        if viz_unnorm == False:
            min_val, max_val = org_x.min(dim=1, keepdim=True)[0], org_x.max(dim=1, keepdim=True)[0]
            org_x = (org_x - min_val) / (max_val - min_val + 1e-8)
            outputs = (outputs - min_val) / (max_val - min_val + 1e-8)
            print(f"[Normalized] sample shape: {org_x.shape}, mask shape: {masks.shape}, output shape: {outputs.shape}")

        # compute its min/max (you can also hard-code 0,1 if you prefer)
        ymin = org_x.min().item()
        ymax = org_x.max().item()
        margin = 0.05 * (ymax - ymin)  # 5% margin

        # split org_x and outputs along time axis
        org_x_splits = torch.split(org_x, viz_split, dim=0)
        outputs_splits = torch.split(outputs, viz_split, dim=0)
        masks_splits = torch.split(masks, viz_split, dim=0)
        print(f"Split into {len(org_x_splits)} segments of length {viz_split} (last segment may be shorter)")

        # loop over splits
        for split in range(len(org_x_splits)):

            org_x_split = org_x_splits[split]
            outputs_split = outputs_splits[split]
            mask_split = masks_splits[split]

            # selected channels to plot
            CH = random.sample(range(org_x.shape[1]), min(viz_channels, org_x.shape[1]))  # select up to viz_channels random channels

            # now loop over channels and plot
            for ch in CH:

                # --- MAE on masked positions only (this channel) ---
                midx = mask_split[:, ch].bool()
                if midx.any():
                    mae = ((outputs_split[midx, ch] - org_x_split[midx, ch])).abs().mean().item()
                    mae_str = f"{mae:.4e}"
                else:
                    mae = float("nan")
                    mae_str = "NA"

                #print(f'==>>> Plotting sample {bidx}, split {split}, channel {ch}, MAE on masked points: {mae_str}')

                # create figure
                fig = plt.figure(figsize=(24, 5))

                #################### ground truth ###########
                plt.plot(
                    org_x_split[:, ch], 
                    marker = '.', 
                    linestyle = '-', 
                    label='Sensor Reading (Ground Truth)', 
                    color='tab:blue',
                    markersize=8,
                    markeredgewidth=1.0,
                    zorder=1
                    )

                ################## masked points #############
                masked_y = org_x_split[:, ch].clone()
                masked_y[~mask_split[:, ch]] = torch.nan  # hide unmasked points
                plt.plot(
                    masked_y, 
                    'x', 
                    label='Masked Points', 
                    fillstyle='none', 
                    markeredgecolor='red',
                    markersize=12, 
                    markeredgewidth=2.0,
                    zorder=2
                    )
                
                ##### predicted values for masked positions #####
                pred_masked_y = outputs_split[:, ch].clone()
                pred_masked_y[~mask_split[:, ch]] = torch.nan  # hide unmasked points
                plt.plot(
                    pred_masked_y, 
                    marker='o', 
                    linestyle='None', 
                    color='tab:green',
                    fillstyle='none', 
                    label='Prediction @ Masked', 
                    markeredgewidth=2.5,
                    markersize=12,
                    zorder=5
                    )

                filename = f"Sample-{bidx}_Split-{split}_Ch-{ch}_M-{model_name}_DS-{dataset_id}_MR-{test_mask_ratio:.2f}_MAE={mae_str}"
                plt.title(filename, fontsize=22)

                plt.xlabel("Time step", fontsize=20)
                plt.ylabel("Value", fontsize=20)

                plt.tick_params(axis='both', which='major', labelsize=16)
                plt.legend(fontsize=20)
                plt.grid(True)

                # **Set the same ylim** for all figures:
                #plt.ylim(ymin - margin, ymax + margin)

                fig_name = os.path.join(output_directory, f"{filename}.png")
                if mae_str != "NA":
                    fig.savefig(fig_name, bbox_inches='tight', dpi = 150)
                #print(f"Figure saved to: {fig_name}")
                plt.close(fig)


# Compatibility alias for the misspelled historical helper.
vizualize_test_samples = visualize_test_samples
