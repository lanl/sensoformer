import argparse
import os

# ---- Different model sizes ----
MODELS = {
    # dims, heads, depth
    'XTi': [192,  4,  2], # Extra Tiny ~ 2.38M
    'Ti': [256,  4,  4], # Tiny ~ 5.80M
    'Mi': [384,  6,  4], # Mini ~ 13.12M
    'S': [512,  8,  8], # Small ~ 35.87M
    'B' : [768, 12,  8], # Base ~ 80.54M
    'L' : [1280, 20, 10], # Large ~ 262.72M
    'XL' : [2048, 32, 10], # Extra Large ~ 672M
    'XXL': [2048, 32, 20], # Extra Extra Large ~ 1.175B
    }

class ArgsConfig:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="device",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = parser
        self._add_args(parser)

        ns = parser.parse_args()
        for key, val in vars(ns).items():
            setattr(self, key, val)

        # post-parse fill-in for tf_params
        if self.tf_params is None:
            self.tf_params = MODELS[self.model_size]

    def _add_args(self, parser: argparse.ArgumentParser):
        # --- Hardware / Parallelization ---
        parser.add_argument('--parallel', type=str, choices=['ddp','no'], default='no', 
                            help='ddp vs no')
        parser.add_argument('--device_idx', type=int, default=0)
        parser.add_argument('--local_rank', type=int, default=int(os.getenv('LOCAL_RANK',0)))
        parser.add_argument('--pin_flag', action='store_true', help="cpu->gpu")
        parser.add_argument('--num_workers', type=int, default=1)

        # --- Data / Dataset Config ---
        parser.add_argument('--seq_len', type=int, default=1024)
        parser.add_argument('--stride', type=int, default=256, 
                            help='less stride -> more overlapping patches, more batches per epoch')
        parser.add_argument('--rp', type = float, default = 0.15, help='for point masking, the fraction of points to mask')
        parser.add_argument('--rt', type = float, default = 0.75, help='for time masking, the fraction of timesteps to mask in the selected row')
        parser.add_argument('--rc', type = float, default = 0.60, help = 'for channel masking, the fraction of channels to mask in the selected column')
        parser.add_argument('--rb', type=float, default = 0.15,
                            help='for block masking, the area ratio of the block to mask')
        parser.add_argument('--sentinel', type=float, default=0.0, help='mask value sentinel')
        parser.add_argument('--keep_data', type=float, default=1.0, help='fraction of training data to keep')

        # --- Model Config ---
        parser.add_argument('--model_size', type=str, choices = list(MODELS.keys()),
                            default='XTi', help='choose from XTi, Ti, Mi, S, B, L, XL, XXL')
        parser.add_argument('--tf_params', nargs=3, type=int,
                            metavar=('d_model','nheads','nlayers'),
                            default=None, help='d_model, nheads, nlayers')
        parser.add_argument('--mslots_decoder', type=int, default=32,
                            help='memory slots in set decoder')
        parser.add_argument('--slots_encoder', type=int, default=16,
                            help='slots in set encoder')
        parser.add_argument('--pool_type', type=str, choices=['mean','gated'],
                            default='gated', help='pooling type in set encoder')
        parser.add_argument('--pos_type', type=str, choices=['learned','absolute','rope'],
                            default='absolute', help='positional encoding type')
        parser.add_argument('--pe_max_len', type=int, default=1024, 
                            help='maximum length for positional encoding')
        

        # --- Training / Optimizer / Schedules ---
        parser.add_argument('--hyperparameter_study', action='store_true',
                            help='if defined, use fixed small settings for quick hyperparameter study')
        parser.add_argument('--flash_attention', action='store_true', 
                            help = 'use flash attention if available')
        parser.add_argument('--amp_dtype', type=str, choices = ['fp16', 'bf16'],
                            default = 'bf16')
        parser.add_argument('--accum_steps', type=int, default=64, help='gradient accumulation steps')
        parser.add_argument('--scale_unmask', type=float, default=0.05,
                            help='loss scaling for unmasked positions')
        parser.add_argument('--scale_smooth', type=float, default=0.01,
                            help='loss scaling for smoothness regularization')
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--num_epochs', type=int, default=25)
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--ckpt_name', type=str, default=None)