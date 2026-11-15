import time
import torch
from torch.nn.utils import clip_grad_norm_
from sensoformer.training.distributed import is_main_process
from sensoformer.data.augmentation import BatchAugmentor

class Trainer:
    @staticmethod
    def train_epoch_amp_ga(model, loader, optimizer, device, 
                            enable_scaler_fp16 = True, amp_dtype = 'fp16',
                            scheduler = None,  accum_steps = 1,  
                            lam_1 = 0.05, lam_2 = 0.01,
                            epoch = 0,
                            hyperparameter_study = False, 
                            run = None,
                            hyper_log_freq = None):          
        """
        AMP training with gradient accumulation.
        'accum_steps' micro-batches are accumulated before each optimizer step.
        Effective batch size = accum_steps * loader.batch_size.
        """
        model.train()
        scaler = torch.amp.GradScaler('cuda', enabled=enable_scaler_fp16)

        window_loss = 0.0
        window_size = 5000
        n_batches   = 0
        accum_count = 0
        batch_losses = []

        masked_sse   = torch.zeros((), device=device)
        masked_cnt   = torch.zeros((), device=device, dtype=torch.long)
        unmasked_sse = torch.zeros((), device=device)
        unmasked_cnt = torch.zeros((), device=device, dtype=torch.long)
        smooth_sum   = torch.zeros((), device=device)
        smooth_cnt   = torch.zeros((), device=device, dtype=torch.long)
        
        start_time  = time.time()

        # IMPORTANT: zero grads once before the loop for accumulation
        optimizer.zero_grad(set_to_none=True)      ### GA: moved out of loop

        # main loop
        for batch in loader:
            x, mask, mask_ratio, mask_type = batch

            x      = x.contiguous().to(device, non_blocking=True)
            mask   = mask.contiguous().to(device, non_blocking=True)       # (B,L,C) bool

            # guard against NaN/Inf in input
            if torch.isnan(x).any() or torch.isinf(x).any():
                continue  # skip batches with NaN/Inf in input

            # norm batch on the fly 
            x, const1, const2, _ = BatchAugmentor.minmaxnorm_01_on_fly(x, mask)
            
            # noisy batch
            x_in, sigma = BatchAugmentor.add_noise(x, mask)

            # mixed-precision forward
            dtype = torch.float16 if amp_dtype == 'fp16' else torch.bfloat16
            with torch.amp.autocast('cuda', dtype=dtype):
                y, z, kv, e_c, q_tokens = model(x_in, mask)

            # guard against NaN/Inf in output (fp16/bf16 can more easily go out of range)
            if not torch.isfinite(y).all():
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0  # reset accumulation
                continue

            # --- Loss in FP32 (outside autocast) for stability ---
            x32 = x.float()
            y32 = y.float()

            sq = (y32 - x32)**2                     # [B, L, C]
            #sq = F.smooth_l1_loss(y32, x32, reduction="none", beta=1.0)

            masked_idx   = mask
            unmasked_idx = ~mask

            # --- masked and unmasked mse loss ---
            m_cnt = masked_idx.sum()                                  # int64
            u_cnt = unmasked_idx.sum()
            m_sse = (sq * masked_idx.to(sq.dtype)).sum()              # float
            u_sse = (sq * unmasked_idx.to(sq.dtype)).sum()

            masked_mse   = m_sse / m_cnt.to(sq.dtype)    # mean for backward
            unmasked_mse = u_sse / u_cnt.to(sq.dtype)
            eff_unmasked_mse = lam_1 * unmasked_mse

            # --- smoothness penalty (time derivative) ---
            del_y = (y32[:, 1:, :] - y32[:, :-1, :]).abs()          # [B, L-1, C]
            s_cnt = del_y.numel()
            s_sum = del_y.sum()
            loss_smooth = s_sum / s_cnt
            eff_loss_smooth = lam_2 * loss_smooth

            # --- overall loss --- 
            loss = masked_mse + eff_unmasked_mse + eff_loss_smooth

            # ---- tripwire for NaN/Inf in loss ----
            if not torch.isfinite(loss):
                if is_main_process():
                    print("\n[NaN/Inf DETECTED in loss]")
                    print("  batch:", n_batches + 1)
                    print("  masked_mse:", masked_mse.item())
                    print("  unmasked_mse:", unmasked_mse.item())
                    print("  loss_smooth:", loss_smooth.item())
                    print("  x min/max:", x.min().item(), x.max().item())
                    print("  y min/max:", y32.min().item(), y32.max().item())
                raise RuntimeError("NaN/Inf in loss")
            
            # --- accumulate batch loss for logging ---
            n_batches   += 1
            window_loss += float(loss.item())
            batch_losses.append(float(loss.item()))

            # ---- logging for hyperparameter study ----
            if (is_main_process() and hyperparameter_study and run is not None
                and hyper_log_freq is not None
                and n_batches % hyper_log_freq == 0):

                global_batch = epoch * len(loader) + n_batches
                run.log({
                    "train/batch_loss": float(loss.item()),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "train/global_batch": global_batch,
                })

            # ---- gradient accumulation ----
            loss_to_backward = loss / accum_steps        ### GA: scaled loss
            scaler.scale(loss_to_backward).backward()    ### GA: accumulate grads
            accum_count += 1

            # ---- accumulate epoch totals (unbiased) ----
            masked_sse   += m_sse.detach()
            masked_cnt   += m_cnt.detach()
            unmasked_sse += u_sse.detach()
            unmasked_cnt += u_cnt.detach()
            smooth_sum   += s_sum.detach()
            smooth_cnt   += s_cnt

            # --- logging is still per micro-batch ---
            # number of masked elements (for logging/debugging)
            num_masked = mask.sum().item()
            if is_main_process():
                if n_batches < 101:
                    elapsed_0 = (time.time() - start_time) / 60
                    print(f'===[Batch={n_batches}, Ct={elapsed_0:.3f}min, '
                          f'Mask type: {mask_type}, '
                          f'Mask ratio: {mask_ratio.float().item():.5f}, '
                          f'Num masked: {num_masked}, '
                          f'Const1: {const1.mean().item():.5f}, Const2: {const2.mean().item():.5f}, '
                          f'x_min: {x.min().item():.5f}, x_max: {x.max().item():.5f}, '
                          f'y_min: {y32.min().item():.5f}, y_max: {y32.max().item():.5f}, '
                          f'Noise level: {sigma:.5f}]===')
                    print(f'→ Inp: {x.shape}, Mask: {mask.shape}, Tar: {y.shape}, Batch loss: {loss.item():.5f}')
                    print(f'→ Eff. Losses: Masked: {masked_mse:.5f}, '
                        f'Unmasked: {eff_unmasked_mse:.5f}, '
                        f'Smooth: {eff_loss_smooth:.5f}')
                    print(f"→ Current LR: {optimizer.param_groups[0]['lr']:.9e}")

                if n_batches % window_size == 0:
                    elapsed = (time.time() - start_time) / 60
                    avg_win = window_loss  / window_size
                    window_loss = 0.0

                    print(f"Ct={elapsed:.2f} min, Batches={n_batches} | "
                        f"Window Avg Loss={avg_win:.5f} | Current LR: {optimizer.param_groups[0]['lr']:.9e}")

            # ---- OPTIMIZER STEP EVERY accum_steps MICRO-BATCHES ----
            if accum_count == accum_steps:                   ### GA: only step periodically
                # unscale + clip before stepping
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                accum_count = 0 # reset accumulation

        # ---- HANDLE REMAINDER ----
        if accum_count > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)
                
        # DDP: sum totals across ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for t in (masked_sse, masked_cnt, unmasked_sse, unmasked_cnt, smooth_sum, smooth_cnt):
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

        # ---- epoch loss ---
        epoch_masked_mse = (masked_sse / masked_cnt.clamp(min=1).to(masked_sse.dtype)).item()
        epoch_unmasked_mse = (unmasked_sse / unmasked_cnt.clamp(min=1).to(unmasked_sse.dtype)).item()
        epoch_smooth = (smooth_sum / smooth_cnt.clamp(min=1).to(smooth_sum.dtype)).item()

        epoch_loss = epoch_masked_mse + lam_1 * epoch_unmasked_mse + lam_2 * epoch_smooth

        # ---- logging for hyperparameter study ---
        if is_main_process() and hyperparameter_study and run is not None:
            run.log({
                "train/epoch_loss": float(epoch_loss),
                "train/epoch": epoch + 1,
            })

        return epoch_loss, batch_losses


    @staticmethod
    def eval_epoch(model, loader, device, epoch = 0, 
                   hyperparameter_study = False, run = None,
                   hyper_log_freq = None):
        model.eval()

        # deterministic eval (only affects torch RNG, not numpy/random inside dataset)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        masked_sse = torch.zeros((), device=device)
        masked_cnt = torch.zeros((), device=device, dtype=torch.long)

        n_batches   = 0
        batch_losses = []

        with torch.no_grad():
            for batch in loader:
                x, mask, mask_ratio, mask_type = batch

                x = x.contiguous().to(device, non_blocking=True)
                mask = mask.contiguous().to(device, non_blocking=True)       

                if torch.isnan(x).any() or torch.isinf(x).any():
                    continue

                # norm batch on the fly 
                x, _, _, _ = BatchAugmentor.minmaxnorm_01_on_fly(x, mask)

                y, z, kv, e_c, q_tokens = model(x, mask)

                masked_idx = mask
                sq = (y - x) ** 2
                #sq = F.smooth_l1_loss(y, x, reduction="none", beta=1.0)

                n_batches   += 1
                batch_masked_cnt = masked_idx.sum()
                batch_masked_sse = (sq * masked_idx.to(sq.dtype)).sum()

                # per-batch eval loss
                batch_loss = (
                    batch_masked_sse /
                    batch_masked_cnt.clamp(min=1).to(batch_masked_sse.dtype)
                ).item()
                batch_losses.append(float(batch_loss))

                # log for hyperparameter study
                if (is_main_process() and hyperparameter_study and run is not None
                    and hyper_log_freq is not None
                    and n_batches % hyper_log_freq == 0):

                    global_batch = epoch * len(loader) + n_batches
                    run.log({
                        "val/batch_loss": float(batch_loss),
                        "val/global_batch": global_batch,
                    })

                masked_cnt += batch_masked_cnt
                masked_sse += batch_masked_sse

        # DDP: one reduce at the end
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(masked_sse, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(masked_cnt, op=torch.distributed.ReduceOp.SUM)
        
        # epoch loss
        epoch_loss = (masked_sse / masked_cnt.clamp(min=1).to(masked_sse.dtype)).item()

        # ---- logging for hyperparameter study ---
        if is_main_process() and hyperparameter_study and run is not None:
            run.log({
                "val/epoch_loss": float(epoch_loss),
                "val/epoch": epoch + 1,
            })

        return epoch_loss, batch_losses
