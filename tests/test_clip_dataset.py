import torch
from training.data_clip_features import ClipFeatureDataset


def test_clip_feature_dataset_shapes():
    ds = ClipFeatureDataset(split="train", num_samples=1)
    patch, feat, label = ds[0]
    assert patch.shape == feat.shape
    assert isinstance(label, int)

