import numpy as np
import torch
from skimage.metrics import structural_similarity

from restoration.losses import L1SSIMLoss, psnr, ssim
from restoration.models.autoencoder import DenoisingAE
from restoration.models.classifier import CorruptionClassifier
from restoration.models.moe import SoftMoE


def test_ssim_matches_skimage_and_identity():
    g = torch.Generator().manual_seed(0)
    a = torch.rand(2, 3, 128, 128, generator=g)
    b = (a + 0.1 * torch.randn(a.shape, generator=g)).clamp(0, 1)
    assert abs(ssim(a, a).item() - 1.0) < 1e-5
    ref = np.mean([structural_similarity(a[i].permute(1, 2, 0).numpy(), b[i].permute(1, 2, 0).numpy(),
                                         data_range=1.0, channel_axis=2, gaussian_weights=True,
                                         sigma=1.5, use_sample_covariance=False) for i in range(2)])
    assert abs(ssim(a, b).item() - ref) < 2e-3


def test_psnr_known_value():
    a = torch.zeros(1, 3, 8, 8)
    b = torch.full_like(a, 0.1)
    assert abs(psnr(a, b).item() - 20.0) < 1e-3


def test_autoencoder_shapes_and_bottleneck():
    m = DenoisingAE(base_ch=16, latent_dim=256)
    x = torch.rand(2, 3, 128, 128)
    assert m.encode(x).shape == (2, 256)
    y = m(x)
    assert y.shape == x.shape and 0 <= y.min() and y.max() <= 1


def test_loss_backward():
    m = DenoisingAE(base_ch=8, latent_dim=64)
    x = torch.rand(2, 3, 128, 128)
    loss, l1, s = L1SSIMLoss(0.8)(m(x), x)
    loss.backward()
    assert torch.isfinite(loss)


def test_classifier_and_moe():
    gate = CorruptionClassifier(8, 3)
    experts = [DenoisingAE(8, 64) for _ in range(3)]
    moe = SoftMoE(gate, experts, tau=1.0).eval()
    x = torch.rand(2, 3, 128, 128)
    recon, logits, w, outs = moe(x)
    assert recon.shape == x.shape and logits.shape == (2, 4) and w.shape == (2, 4)
    assert torch.allclose(w.sum(1), torch.ones(2), atol=1e-5)
    manual = w[:, 0, None, None, None] * x + sum(w[:, k + 1, None, None, None] * outs[:, k] for k in range(3))
    assert torch.allclose(recon, manual, atol=1e-5)


def test_spatial_latent_autoencoder_shapes():
    m = DenoisingAE(base_ch=8, latent_type="spatial", latent_ch=8)
    x = torch.rand(2, 3, 128, 128)
    z = m.encode(x)
    assert z.shape == (2, 8, 8, 8) and m.latent_dim == 8 * 64  # 512 numbers vs 49152 input values
    y = m(x)
    assert y.shape == x.shape and 0 <= y.min() and y.max() <= 1
