import torch

from sensoformer import SensOFormer


def _model():
    return SensOFormer(
        seq_len=9,
        dims=16,
        head=4,
        mlp_dim=32,
        depth=2,
        dropout=0.0,
        slots_encoder=3,
        mslots_decoder=2,
        pos_type="absolute",
    ).eval()


def test_encoder_permutation_invariance():
    torch.manual_seed(21)
    model = _model()
    x = torch.randn(2, 9, 6)
    mask = torch.rand_like(x) < 0.2
    permutation = torch.randperm(x.shape[-1])

    with torch.no_grad():
        original = model.encode(x, mask)
        permuted = model.encode(x[..., permutation], mask[..., permutation])

    torch.testing.assert_close(permuted, original, rtol=1e-5, atol=1e-6)


def test_decoder_permutation_equivariance():
    torch.manual_seed(22)
    model = _model()
    x = torch.randn(2, 9, 6)
    mask = torch.rand_like(x) < 0.2
    permutation = torch.randperm(x.shape[-1])

    with torch.no_grad():
        original = model(x, mask)[0]
        permuted = model(x[..., permutation], mask[..., permutation])[0]

    torch.testing.assert_close(
        permuted, original[..., permutation], rtol=1e-5, atol=1e-6
    )
