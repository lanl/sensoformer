![SensOFormer: Foundation Models for Particle Accelerators](docs/assets/sensoformer.gif)

<h1 align="center">SensOFormer: Foundation Models for Particle Accelerators</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2611.02031"><img src="https://img.shields.io/badge/arXiv-2611.02031-b31b1b" alt="Paper on arXiv"></a>
  <a href="docs/assets/neurips26_paper_sensoformer.pdf"><img src="https://img.shields.io/badge/Paper-PDF-blue" alt="Paper PDF"></a>
  <a href="https://mahindrautela.github.io/sensoformer/"><img src="https://img.shields.io/badge/Project_Website-SensOFormer-blue" alt="SensOFormer project website"></a>
</p>

SensOFormer is a denoising masked transformer for multivariate sensor signals. A
Perceiver-style set encoder and decoder handle variable sensor sets while a compact
temporal latent models their dynamics.

## Release scope

- This repository contains the SensOFormer architecture, masking and noise augmentation, HDF5 sequence loading, self-supervised pretraining, zero-shot missing-data imputation, metrics, optional visualization, DDP support, tests, and a synthetic example.

- Task-specific fine-tuning pipelines, anomaly-detection and fault-identification heads, research notebooks, generated results, datasets, and pretrained weights are not included. Fine-tuning scripts for three downstream tasks—anomaly detection, cavity identification, and fault identification—are planned for release in the coming weeks.

- Model checkpoints will be coming soon on hugging face.

- Datasets will take time to pass through LANL release.


## Installation

Python 3.11 or newer is required.

```bash
pip install -e .
```

Install optional features with `pip install -e ".[visualization]"` or
`pip install -e ".[wandb]"`. Development tools are available through
`pip install -e ".[dev]"`.

## Minimal use

```python
import torch
from sensoformer import SensOFormer

model = SensOFormer(
    seq_len=32, dims=32, head=4, mlp_dim=64, depth=2, dropout=0.0,
    slots_encoder=2, mslots_decoder=2, pos_type="absolute",
)
x = torch.randn(1, 32, 8)
mask = torch.rand_like(x) < 0.15
reconstruction, latent, *_ = model(x, mask)
```

Run the dataset-free CPU example with `python examples/synthetic_forward.py`.

## Pretraining and inference

From the repository root, after installation:

```bash
python -m scripts.pretrain --model_size XTi --seq_len 1024
python -m scripts.inference --checkpoint CHECKPOINT.pth --ds 0 --viz_batches 0
```

The scripts use the dataset locations in `config/config_dataset.py`. W&B is imported only when `--hyperparameter_study` is enabled.

## HDF5 input

Each file must contain a `data` group whose children are two-dimensional numeric datasets:

```text
/data/0  shape: (time, channels)
/data/1  shape: (time, channels)
...
```

See [DATA.md](DATA.md) for details. Datasets and checkpoints are not distributed here;
Hugging Face artifacts are coming soon.

## Citation

If this code supports your work, cite *SensOFormer: Foundation Models for Particle Accelerators* by Mahindra Rautela and Alexander Scheinker (arXiv:2611.02031). Citation is provided in [CITATION.cff](CITATION.cff).

#### Note: EIDR number O#5133 - SensOFormer. This program is Open-Source under the MIT License.
