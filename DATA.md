# Data format

SensOFormer expects HDF5 files with a top-level `data` group. Every child of that group is
one sequence, stored as a numeric array with shape `(time, channels)`. Keys may be integer
strings (`0`, `1`, ...) or other sortable strings. All sequences within a training batch
must have compatible channel counts.

The loader creates sliding windows using `seq_len` and `stride`, converts samples to
`float32`, and generates a point, time, channel, or block mask. Training and validation are
split at the sequence level to avoid sharing windows from one sequence across splits.

No raw or processed datasets are distributed in this repository. Dataset paths remain
site-specific configuration. Public datasets may be provided separately on Hugging Face
in the future; they are coming soon.
