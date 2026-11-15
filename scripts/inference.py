"""Zero-shot missing-data imputation with a pretrained SensOFormer checkpoint."""

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config.config_dataset import DataConfig
from sensoformer import SensOFormer
from sensoformer.inference.data import generate_missing_data, get_test_data
from sensoformer.inference.evaluate import test_model
from sensoformer.training.devices import DeviceManager


MODELS = {
    "XTi": [192, 4, 2], "Ti": [256, 4, 4], "Mi": [384, 6, 4],
    "S": [512, 8, 8], "B": [768, 12, 8], "L": [1280, 20, 10],
    "XL": [2048, 32, 10], "XXL": [2048, 32, 20],
}

INFERENCE_KEYS = [
    "LANSCE_dt_1", "LANSCE_dt_2", "LHC_dt", "HVCM_SNS_dt",
    "LCLS_SLAC_dt", "SRF_CEBAF_dt", "RCS_Fermilab_dt", "TCV_tokamak_dt",
]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--hyperparameter_study", action="store_true")
    parser.add_argument(
        "--test_mask_ratio", type=float, default=0.05,
        choices=[0.05, 0.15, 0.25, 0.50],
    )
    parser.add_argument(
        "--test_mask_type", default="point",
        choices=["point", "time", "channel", "block"],
    )
    parser.add_argument("--ds", type=int, default=0, choices=range(8))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run_batches", type=int, default=None)
    parser.add_argument("--viz_batches", type=int, default=25)
    parser.add_argument("--num_instances", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=1234)
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    std = array.std(ddof=1) if len(array) > 1 else 0.0
    return float(array.mean()), float(std)


def main():
    args = build_parser().parse_args()
    if args.num_instances < 1:
        raise ValueError("--num_instances must be >= 1")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    project_root = Path(__file__).resolve().parents[1]

    print("===== Arguments Used =====")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    run = None
    if args.hyperparameter_study:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging was enabled, but wandb is not installed. "
                "Install sensoformer[wandb]."
            ) from exc
        run = wandb.init(config=vars(args))

    devices = DeviceManager.list_devices()
    if args.device_idx >= len(devices):
        raise ValueError(
            f"--device_idx={args.device_idx} is unavailable; found {len(devices)} device(s)"
        )
    device = devices[args.device_idx]

    checkpoint_path = project_root / "models" / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["args"]
    model_size = config["model_size"]
    d_model, nheads, nlayers = MODELS[model_size]
    seq_len = config["seq_len"]
    model = SensOFormer(
        seq_len=seq_len,
        dims=d_model,
        head=nheads,
        mlp_dim=4 * d_model,
        depth=nlayers,
        dropout=0.05,
        slots_encoder=config["slots_encoder"],
        mslots_decoder=config["mslots_decoder"],
        pool_type=config["pool_type"],
        SENTINEL=config["sentinel"],
        pos_type=config["pos_type"],
        pe_max_len=config["pe_max_len"],
    ).to(device)

    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.eval()
    print(f"NUM PARAMS (in M): {sum(p.numel() for p in model.parameters()) / 1e6:.2f}")

    inference_data = INFERENCE_KEYS[args.ds]
    h5_paths, _ = DataConfig(str(project_root)).collect_h5_paths([inference_data])
    h5_path = h5_paths[0]
    print(f"Selected HDF5 file for inference: {h5_path}")

    output_directory = (
        project_root / "experiments" / "results" / "imputation"
        / datetime.now().strftime("%Y-%m-%d") / inference_data / model_size
        / f"masktype-{args.test_mask_type}" / f"maskratio-{args.test_mask_ratio:.2f}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    test_data_list = get_test_data(h5_path)

    metric_names = ("mae", "mse", "mre", "rel_l2", "nmae_range_pct", "mape")
    aggregate = {name: [] for name in metric_names}
    last_details = None
    last_visuals = None

    for instance_idx in range(args.num_instances):
        seed = args.base_seed + instance_idx
        set_seed(seed)
        print(f"Inference instance {instance_idx + 1}/{args.num_instances}; seed={seed}")
        missing_data = generate_missing_data(
            test_data_list,
            mask_ratio=args.test_mask_ratio,
            mask_type=args.test_mask_type,
            output_directory=str(output_directory),
            plot_example=not args.hyperparameter_study and args.viz_batches > 0,
        )
        result = test_model(
            model, test_data_list, missing_data, seq_len=seq_len,
            device=device, run_batches=args.run_batches,
        )
        last_visuals = result[0]
        last_details = result[1:7]
        for name, value in zip(metric_names, result[7:13]):
            aggregate[name].append(float(value))

    summary = {name: _mean_std(values) for name, values in aggregate.items()}
    print("===== Mean/std over inference instances =====")
    for name, (mean, std) in summary.items():
        print(f"{name}: {mean:.6f} ± {std:.6f}")

    metrics_path = output_directory / "masked_errors_test.txt"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for name, (mean, std) in summary.items():
            handle.write(f"{name}: {mean:.8f} ± {std:.8f}\n")
        handle.write("Per-example metrics from the last inference instance:\n")
        for name, values in zip(metric_names, last_details):
            for index, value in enumerate(values):
                handle.write(f"example {index}, {name}: {value:.8f}\n")

    n_viz = min(args.viz_batches, len(last_visuals))
    if not args.hyperparameter_study and n_viz > 0:
        try:
            from sensoformer.inference.visualization import visualize_test_samples
        except ImportError as exc:
            raise RuntimeError(
                "Visualization requires matplotlib. Install sensoformer[visualization] "
                "or pass --viz_batches 0."
            ) from exc
        visualize_test_samples(
            last_visuals, str(output_directory),
            test_mask_ratio=args.test_mask_ratio, viz_channels=10,
            viz_batches=n_viz, model_name=model_size, dataset_id=args.ds,
        )

    if run is not None:
        run.log({
            "num_instances": args.num_instances,
            **{
                f"avg_masked_{name}_{stat}": value[index]
                for name, value in summary.items()
                for index, stat in enumerate(("mean", "std"))
            },
        })
        run.finish()


if __name__ == "__main__":
    main()
