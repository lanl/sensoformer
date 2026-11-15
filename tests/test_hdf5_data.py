import h5py
import numpy as np

from sensoformer.data.hdf5_dataset import SequenceDataset
from sensoformer.data.loaders import DataLoaders


def _write_hdf5(path):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        for index in range(3):
            values = np.arange(72, dtype=np.float32).reshape(24, 3) + index
            group.create_dataset(str(index), data=values)


def _close_cached_files():
    for handle in SequenceDataset._FILES.values():
        handle.close()
    SequenceDataset._FILES.clear()


def test_sequence_dataset_and_loaders(tmp_path):
    path = tmp_path / "synthetic.h5"
    _write_hdf5(path)
    try:
        dataset = SequenceDataset((path, "0"), seq_len=8, stride=4)
        x, mask, ratio, mask_type = dataset[0]
        assert x.shape == mask.shape == (8, 3)
        assert mask_type in {"point", "time", "channel", "block"}
        assert 0.0 < ratio < 1.0

        train, validation = DataLoaders(
            [path], seq_len=8, stride=8, batch_size=2,
            num_workers=0, pin_memory=False, seed=1,
        )
        batch_x, batch_mask, _, _ = next(iter(train))
        assert batch_x.shape == batch_mask.shape
        assert batch_x.shape[1:] == (8, 3)
        assert len(validation) > 0
    finally:
        _close_cached_files()
