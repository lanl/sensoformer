"""Small, dataset-free SensOFormer CPU forward pass."""

import torch

from sensoformer import SensOFormer


def main():
    torch.manual_seed(7)
    model = SensOFormer(
        seq_len=12,
        dims=16,
        head=4,
        mlp_dim=32,
        depth=1,
        dropout=0.0,
        slots_encoder=2,
        mslots_decoder=2,
        pos_type="absolute",
    ).cpu().eval()
    x = torch.randn(2, 12, 5)
    mask = torch.rand_like(x) < 0.2

    with torch.no_grad():
        reconstruction, latent, *_ = model(x, mask)

    print(f"input: {tuple(x.shape)}")
    print(f"reconstruction: {tuple(reconstruction.shape)}")
    print(f"latent: {tuple(latent.shape)}")
    print("Synthetic inputs exercise the API only; they do not demonstrate prediction quality.")


if __name__ == "__main__":
    main()
