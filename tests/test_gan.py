import numpy as np
import torch

from restoration.data import fs2k
from restoration.models.gan import PatchDiscriminator, UNetGenerator


def test_generator_shape_range_and_style_matters():
    G = UNetGenerator(base_ch=8, style_dim=8, dropout=0.0).eval()
    x = torch.rand(2, 3, 128, 128) * 2 - 1
    y0 = G(x, torch.tensor([0, 0]))
    y1 = G(x, torch.tensor([1, 2]))
    assert y0.shape == (2, 3, 128, 128) and y0.abs().max() <= 1
    assert not torch.allclose(y0, y1, atol=1e-6)  # the embedding actually changes the output


def test_discriminator_patch_output_and_style_matters():
    D = PatchDiscriminator(base_ch=8, style_dim=8).eval()
    x, y = torch.rand(2, 3, 128, 128), torch.rand(2, 3, 128, 128)
    o0, o1 = D(x, y, torch.tensor([0, 0])), D(x, y, torch.tensor([1, 2]))
    assert o0.shape == (2, 1, 14, 14)
    assert not torch.allclose(o0, o1, atol=1e-6)


def test_style_embedding_receives_gradients():
    G = UNetGenerator(base_ch=8, style_dim=8, dropout=0.0)
    out = G(torch.rand(2, 3, 128, 128), torch.tensor([0, 1]))
    out.mean().backward()
    assert G.embed.weight.grad.abs().sum() > 0


def test_split_stratified_sizes_and_disjoint():
    s = fs2k.split_indices()
    assert abs(len(s["val"]) / (len(s["val"]) + len(s["train"])) - 0.15) < 0.005
    assert not set(s["train"]) & set(s["val"])
    _, _, st = fs2k._cache("train")
    for k in range(3):
        frac = (st[s["val"]] == k).sum() / (st == k).sum()
        assert abs(frac - 0.15) < 0.02  # stratified by style


def test_paired_augmentation_keeps_alignment():
    ds = fs2k.FS2KDataset("train")
    # make photo and sketch identical so any unpaired transform would make x != y
    ds.sketches = ds.photos
    for i in range(40):
        x, y, _ = ds[i]
        assert torch.allclose(x, y, atol=1e-5)


def test_official_test_is_separate():
    tr, te = fs2k._cache("train"), fs2k._cache("test")
    assert len(tr[2]) == 1058 and len(te[2]) == 1046
    assert len(fs2k.FS2KDataset("test")) == 1046
