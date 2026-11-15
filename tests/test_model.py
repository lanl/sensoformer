import torch

from sensoformer import SensOFormer


def make_model(seq_len=10):
    return SensOFormer(
        seq_len=seq_len,
        dims=16,
        head=4,
        mlp_dim=32,
        depth=1,
        dropout=0.0,
        slots_encoder=2,
        mslots_decoder=2,
        pos_type="absolute",
    ).eval()


def test_forward_shape_and_encode():
    model = make_model()
    x = torch.randn(2, 10, 5)
    mask = torch.rand_like(x) < 0.2

    with torch.no_grad():
        output, latent, kv, context, queries = model(x, mask)

    assert output.shape == x.shape
    assert latent.shape == (2, 10, 16)
    assert kv.shape == queries.shape == (2, 10, 5, 16)
    assert context.shape == (2, 5, 16)
    torch.testing.assert_close(model.encode(x, mask), latent)


def test_multiple_channel_counts():
    model = make_model(seq_len=7)
    for channels in (1, 3, 11):
        x = torch.randn(2, 7, channels)
        mask = torch.zeros_like(x, dtype=torch.bool)
        assert model(x, mask)[0].shape == x.shape


def test_state_dict_round_trip(tmp_path):
    torch.manual_seed(9)
    model = make_model()
    x = torch.randn(1, 10, 4)
    mask = torch.rand_like(x) < 0.25
    expected = model(x, mask)[0]
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)

    restored = make_model()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    actual = restored(x, mask)[0]

    torch.testing.assert_close(actual, expected)
    assert set(restored.state_dict()) == set(model.state_dict())
