# SensOFormer model card

## Architecture

SensOFormer embeds each observed value together with its mask bit. A Perceiver-style set
encoder maps an unordered, variable-size channel set to a temporal latent sequence. A
transformer processes time dependencies, and a set decoder reconstructs the requested
channels. Channel-set encoding is permutation invariant and decoding is permutation
equivariant.

## Intended use

The released code supports self-supervised denoising pretraining and zero-shot imputation
of missing multichannel accelerator sensor readings. It is a research implementation, not
a safety or control system. Task-specific fine-tuning pipelines are not included in the
current release. Fine-tuning scripts for anomaly detection, cavity identification, and
fault identification are planned for release in the coming weeks.

## Inputs and outputs

Inputs are a floating-point tensor `(batch, time, channels)` and a Boolean mask of the same
shape, where `True` marks hidden values. The main output is a reconstruction of the same
shape. `encode()` returns the processed temporal latent `(batch, time, hidden_dim)`.

## Limitations

Results depend on training data coverage, normalization, masking pattern, sensor quality,
and distribution shift. Reconstructions may be inaccurate or physically inconsistent and
must not be treated as verified measurements. Performance should be validated for every
facility and operating regime.

Pretrained weights are not distributed in this repository. Hugging Face weights are
planned for the future and are coming soon.
