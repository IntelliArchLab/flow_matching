import torch
from models.clip_one_layer import ClipOneLayer


def test_clip_one_layer_forward():
    model = ClipOneLayer()
    x = torch.randn(2, 10, 768)
    t = torch.rand(2)
    out = model(x, t)
    assert out.shape == x.shape

