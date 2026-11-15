import gc
import math
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from config.argument_parser import ArgsConfig
from config.config_dataset import DataConfig
from sensoformer import SensOFormer
from sensoformer.data.loaders import DataLoaders
from sensoformer.training.devices import DeviceManager
from sensoformer.training.distributed import is_main_process
from sensoformer.training.schedulers import LRSched
from sensoformer.training.trainer import Trainer
    
def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    project_root = str(Path(__file__).resolve().parents[1])
    savepath_logs = os.path.join(project_root, "experiments", "logs")
    os.makedirs(savepath_logs, exist_ok=True)

    args = ArgsConfig()
    data_cfg = DataConfig(project_root)

    # args dictionary for saving
    args_dict = {k: v for k, v in vars(args).items() if k != "parser"}

    # run tags for modelname
    run_tag = (
            f"{args.model_size}_"
            f"SL-{args.seq_len}_"
            f"SR-{args.stride}_"
            f"ES-{args.slots_encoder}_"
            f"MS-{args.mslots_decoder}_"
            f"PE-{args.pos_type}_"
            f"PEML-{args.pe_max_len}_"
            f"DF-{args.keep_data}_"
            f"LR-{args.lr}_"
            f"MR-p{args.rp}-t{args.rt}-c{args.rc}-b{args.rb}_"
            f"EP-{args.num_epochs}"
    )

    # --- load data h5 path ---
    pretrain_keys = ["LANSCE_pt","LHC_pt", "HVCM_SNS_pt", "LCLS_SLAC_pt", "SRF_CEBAF_pt"]
    h5_paths, per_dataset_counts = data_cfg.collect_h5_paths(pretrain_keys)

    # -- model parameters --
    d_model, nheads, nlayers = args.tf_params
    batchsize = 1 # code works only with bs = 1 for now

    # --- set up gpu ----
    if args.parallel == 'ddp':
        if platform.system() == 'Windows':
            ddp_backend = 'gloo'
        else:
            ddp_backend = 'nccl' # pick the right backend

        local_rank = int(os.environ["LOCAL_RANK"])
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=ddp_backend, init_method='env://',
                                rank=rank, world_size=world_size)
        dist.barrier(device_ids=[local_rank]) # wait for everyone
        device = torch.device(f'cuda:{local_rank}')
        print(f"[rank={rank} | local_rank={local_rank} | world_size={world_size}] "
            f"host={socket.gethostname()} current_device={torch.cuda.current_device()}")
    
    else:
        device = torch.device(f'cuda:{args.device_idx}' if torch.cuda.is_available() else 'cpu')

    # print devices, args, and dataset info only on the main process
    if is_main_process():
        devices = DeviceManager.list_devices()

        # print and args
        print("===== Arguments Used =====")
        for k, v in args_dict.items():
            print(f"{k}: {v}")
        print("==========================")

        print("Pretraining datasets:")
        for key, count in per_dataset_counts.items():
            print(f"{key}: {count} file(s)")
        print(f"Total HDF5 files: {len(h5_paths)}")
        print("==========================")

    # --- hyperparameter study ---
    run = None
    if is_main_process() and args.hyperparameter_study:
        print("→ Hyperparameter study mode ON: using small settings for quick runs")
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging was enabled, but wandb is not installed. "
                "Install sensoformer[wandb]."
            ) from exc
        run = wandb.init(config=args_dict)
        
    #breakpoint()
    ###########################################################################
    ####################### Masked transformer model ##########################
    ###########################################################################
    
    modelname = (f"MAINRUN_DMT-{run_tag}") if not args.hyperparameter_study else f"DMT-{run_tag}"

    model = SensOFormer(args.seq_len, dims = d_model,
            head = nheads, mlp_dim = 4 * d_model, depth = nlayers, dropout = 0.05,
            slots_encoder=args.slots_encoder, 
            mslots_decoder= args.mslots_decoder,
            pool_type=args.pool_type,
            SENTINEL = args.sentinel,
            lora_r_attn = 0, lora_r_mlp = 0, 
            lora_alpha = None, lora_p = 0.0, 
            flash_attention=args.flash_attention,
            pos_type = args.pos_type,
            pe_max_len = args.pe_max_len).to(device)
    
    # Select models to run
    savepath_model = os.path.join(project_root, 'models', f'{args.model_size}')
    os.makedirs(savepath_model, exist_ok=True)
    fullmodelname = os.path.join(savepath_model, modelname)
    if is_main_process():
        print(f"NUM PARAMS (in M): {sum(p.numel()/10**6 for p in model.parameters()):.2f}")
        print('Model architecture', model)
    
    #breakpoint()
    ############################################################################
    ############################## parallelization #############################
    ############################################################################

    if args.parallel == 'ddp':
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_gpus = torch.cuda.device_count()
        
        # Gather hostnames from all ranks
        host = socket.gethostname()
        host_list = [None] * world_size
        dist.all_gather_object(host_list, host)
        
        if is_main_process():
            num_nodes = len(set(host_list))
            print(f"\n → DDP on {world_size} GPUs across {num_nodes} node(s) \t")
        
        # Print per-rank info from every rank
        print(f"[Rank {rank}/{world_size}] "
              f"Host={host}, node has {local_gpus} GPUs, "
              f"using cuda:{local_rank}")
        
        dist.barrier(device_ids=[local_rank])
        model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], 
                output_device = local_rank)
    
    elif args.parallel == 'no' :
        world_size, rank, local_rank = 1, 0, 0
        print('→ Single-GPU mode, running unwrapped')

    ############################################################################
    ############################## dataloaders #################################
    ############################################################################

    # get the dataloaders
    train_loader, val_loader = DataLoaders(h5_paths, 
        seq_len = args.seq_len,
        stride = args.stride, 
        mask_ratios = [args.rp, args.rt, args.rc, args.rb],
        batch_size = batchsize, 
        keep_data=args.keep_data, 
        num_workers = args.num_workers, 
        pin_memory = args.pin_flag)
    
    ############################################################################
    ############################## training dynamics ###########################
    ############################################################################

    # gradient accumulation steps
    if world_size > 1:
        accum_steps = max(1, int(args.accum_steps // world_size)) 
    else:
        accum_steps = args.accum_steps

    # number of optimizer updates per epoch (not micro-batches)
    gradient_updates_per_epoch = math.ceil(len(train_loader) / accum_steps)

    # 2.5% of updates for warmup, rounded up to nearest integer
    args.warmup_steps = max(1, math.ceil(0.025 * gradient_updates_per_epoch))

    # total training steps
    args.total_train_steps = args.num_epochs * gradient_updates_per_epoch

    # optional, useful for logs/checkpoints
    args_dict["warmup_steps"] = args.warmup_steps
    args_dict["total_train_steps"] = args.total_train_steps

    if is_main_process():
        print("==========================")
        print(f"Using AMP {args.amp_dtype} with GA of {accum_steps} steps")
        print(f"Optimizer updates/epoch: {gradient_updates_per_epoch}")
        print(f"Warmup steps: {args.warmup_steps}")
        print(f"Total train steps: {args.total_train_steps}")
        print("==========================")

    ##########################################################################
    ########################### Loss and optimizers ##########################
    ##########################################################################
    
    optimizer   = torch.optim.AdamW(model.parameters(), lr = args.lr)
    scheduler = LRSched.warmup_cosine(optimizer, args)

    ##########################################################################
    ############################# Resume logic ###############################
    ##########################################################################
    
    start_epoch = 0
    if args.resume and args.ckpt_name is None:
        raise ValueError("--resume requires --ckpt_name <CHECKPOINT>")
    
    if args.ckpt_name:
        resume_path = os.path.join(savepath_model, args.ckpt_name)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
            
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        if state_dict is None:
            raise KeyError("Checkpoint missing model_state_dict/state_dict")
        
        # normalize keys (strip "module." only where present)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        
        # Pick the real model if wrapped
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            target = model.module 
        else: 
            target = model
        
        # Load weights
        missing, unexpected = target.load_state_dict(state_dict, strict=False)
        if is_main_process() and (missing or unexpected):
            print(f"Warning: missing keys: {missing} | unexpected keys: {unexpected}")
            
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        if is_main_process():
            print(f"Resumed from {resume_path}, starting at epoch {start_epoch}")
    
    ###########################################################################
    ################################# Training ################################
    ###########################################################################
    
    # free up any leftover memory from previous runs
    gc.collect()
    torch.cuda.empty_cache()
        
    train_losses = []
    val_losses   = []
    
    history = {
    "epoch": [],
    "train_loss_epoch": [],
    "val_loss_epoch": [],
    "lr": [],
    "wall_time_s": [],
    "train_batch_loss": [],  # batch-level
    "val_batch_loss": [],    # batch-level
    }
    
    if is_main_process(): 
        print(f"Starting pretraining for {args.num_epochs} epochs...")

    for epoch in range(start_epoch, args.num_epochs):    
        start = time.time()
        # --- make workers deterministic ----
        if dist.is_initialized():
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
        
        # --- training ---    
        tr_epoch_loss, tr_batch_losses = Trainer.train_epoch_amp_ga(model, train_loader, optimizer,
                device, enable_scaler_fp16 = True, amp_dtype = args.amp_dtype,
                scheduler = scheduler, accum_steps = accum_steps,
                lam_1 = args.scale_unmask, lam_2 = args.scale_smooth,
                epoch = epoch,
                hyperparameter_study = args.hyperparameter_study, 
                run = run if is_main_process() and args.hyperparameter_study else None,
                hyper_log_freq=50)

        # --- validation ---
        va, va_batch_losses = Trainer.eval_epoch(model, val_loader, device,
                epoch = epoch, hyperparameter_study = args.hyperparameter_study, 
                run = run if is_main_process() and args.hyperparameter_study else None,
                hyper_log_freq = 50)
        
        epoch_time = (time.time() - start) / 60

        # --- print epoch info ---
        if is_main_process():
            print(f"--> ep{epoch+1}: tpe: {epoch_time:.3f} min, "
                  f"train={tr_epoch_loss:.5f}, val={va:.5f}")

        # --- store the losses ---
        train_losses.append(tr_epoch_loss)
        val_losses.append(va)
        
        # --- store training logging info ---
        history["epoch"].append(epoch)
        history["train_loss_epoch"].append(float(tr_epoch_loss))
        history["val_loss_epoch"].append(float(va))
        history["wall_time_s"].append(float(epoch_time))
        history["train_batch_loss"].extend(tr_batch_losses)
        history["val_batch_loss"].extend(va_batch_losses)

        # -- save checkpoint after every epoch ---
        if is_main_process():
            checkpoint = {"epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "args": args_dict,
                            "history": history}
            ckpt_path = f"{fullmodelname}_ep{epoch+1}.pth" 
            torch.save(checkpoint, ckpt_path)
            print(f" Saved checkpoint: {ckpt_path}")
        
        # free up any leftover memory before the next epoch
        gc.collect()
        torch.cuda.empty_cache()
            
    # --- Plot training & validation loss ---
    if is_main_process():
        try:
            import matplotlib

            if not os.environ.get("DISPLAY"):
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(8,4))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses,   label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training vs Validation Loss')
            plt.legend()
            plt.grid(True)
            fig_name = os.path.join(savepath_logs, f'learning_curve_{modelname}.png')
            fig.savefig(fig_name, bbox_inches='tight')
            print(f"Figure saved to: {fig_name}")
            plt.close(fig)
        except ImportError:
            print("Matplotlib is not installed; skipping the learning-curve plot.")

        # save train & val losses
        losses_path = os.path.join(savepath_logs, f'losses_{modelname}.npz')
        np.savez(losses_path,
                train_losses=np.array(train_losses, dtype=np.float32),
                val_losses=np.array(val_losses, dtype=np.float32))

    # complete the hyperparameter study run
    if is_main_process() and args.hyperparameter_study:
        run.finish()

    # --- clean up DDP ----
    if args.parallel == 'ddp':
        dist.destroy_process_group()
        
if __name__ == "__main__":
    main()
