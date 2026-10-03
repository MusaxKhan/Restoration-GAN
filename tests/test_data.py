import numpy as np
import pytest
import torch

from restoration.data import corruptions as C
from restoration.data import pets


def test_split_is_80_20_disjoint_and_deterministic():
    s = pets.get_split_indices()
    assert len(s["train"]) == 2944 and len(s["val"]) == 736
    assert not set(s["train"]) & set(s["val"])
    assert s == pets.get_split_indices()


def test_salt_pepper_fraction_and_values():
    img = np.full((128, 128, 3), 0.5, np.float32)
    out = C.salt_and_pepper(img, 0.10, seed=1)
    changed = (out != 0.5).any(-1)
    assert abs(changed.mean() - 0.10) < 0.01
    assert set(np.unique(out[changed])) <= {0.0, 1.0}
    assert abs((out[changed] == 1.0).mean() - 0.5) < 0.05  # black/white equally likely


def test_blur_smooths_image():
    rng = np.random.default_rng(0)
    img = rng.random((128, 128, 3)).astype(np.float32)
    out = C.gaussian_blur(img, 5, 1.5)
    assert out.shape == img.shape and out.std() < img.std()


@pytest.mark.parametrize("n,cov", C.TEST_OCC)
def test_occlusion_coverage_close_to_target(n, cov):
    for seed in range(20):
        rects = C.make_rects(np.random.default_rng(seed), n, cov)
        assert len(rects) == n
        assert abs(C.coverage_of(rects) - cov) < 0.04


def test_train_spec_ranges():
    rng = np.random.default_rng(0)
    counts = np.zeros(4)
    for _ in range(4000):
        s = C.sample_train_spec(rng)
        counts[s["type"]] += 1
        if s["type"] == C.SALT:
            assert 0.02 <= s["p"] <= 0.15
        if s["type"] == C.BLUR:
            assert s["k"] in (3, 5, 7) and 0.5 <= s["sigma"] <= 2.5
        if s["type"] == C.OCC:
            assert 1 <= len(s["rects"]) <= 3
            assert 0.08 <= s["coverage"] <= 0.37
    assert np.all(np.abs(counts / 4000 - 0.25) < 0.03)


def test_manifests_deterministic_and_complete():
    assert pets.build_val_manifest() == pets.build_val_manifest()
    t = pets.build_test_manifest()
    assert len(t) == 3669 * 10
    assert t == pets.build_test_manifest()
    levels = {(e["type"], e["level"]) for e in t}
    assert len(levels) == 1 + 9


def test_train_dataset_is_dynamic_and_shapes():
    ds = pets.PetsDataset("train")
    a = ds[0]
    assert a[0].shape == (3, 128, 128) and a[1].shape == (3, 128, 128)
    assert 0.0 <= a[0].min() and a[0].max() <= 1.0
    # same index, repeated calls -> corruption is re-sampled (not a fixed saved copy)
    labels = {ds[0][2] for _ in range(30)}
    assert len(labels) > 1


def test_val_dataset_deterministic():
    ds = pets.PetsDataset("val")
    x1, y1, l1, _ = ds[3]
    x2, y2, l2, _ = ds[3]
    assert torch.equal(x1, x2) and l1 == l2


def test_balanced_sampler_balanced():
    sampler = pets.BalancedBatchSampler(1000, 32)
    batch = next(iter(sampler))
    labels = [c for _, c in batch]
    assert [labels.count(k) for k in range(4)] == [8, 8, 8, 8]
