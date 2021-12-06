import os

import numpy as np
import pytest
import tensorflow as tf
import timm
import torch

from tfimm import create_model, list_models
from tfimm.utils.timm import load_pytorch_weights_in_tf2_model

# Exclude models that cause specific test failures
if "GITHUB_ACTIONS" in os.environ:  # and 'Linux' in platform.system():
    EXCLUDE_FILTERS = ["vit_large_*", "vit_huge_*", "cait_m*", "ig_resnext101_32x48d"]
else:
    EXCLUDE_FILTERS = ["cait_m*"]


@pytest.mark.parametrize("model_name", list_models(exclude_filters=EXCLUDE_FILTERS))
def test_create_model(model_name: str):
    """Test if we can instantiate a model and run a forward pass"""
    model = create_model(model_name)
    model(model.dummy_inputs)


@pytest.mark.parametrize(
    "model_name",
    list_models(
        pretrained="timm",
        exclude_filters=EXCLUDE_FILTERS,
    ),
)
@pytest.mark.timeout(60)
def test_load_timm_model(model_name: str):
    """Test if we can load models from timm."""
    # We don't need to load the pretrained weights from timm, we only need a PyTorch
    # model, that we then convert to tensorflow. This allows us to run these tests
    # in GitHub CI without data transfer issues.
    pt_model = timm.create_model(model_name, pretrained=False)
    pt_model.eval()

    tf_model = create_model(model_name, pretrained=False)
    tf_model = load_pytorch_weights_in_tf2_model(tf_model, pt_model.state_dict())

    rng = np.random.default_rng(2021)
    img = rng.random(
        size=(1, *tf_model.cfg.input_size, tf_model.cfg.in_chans), dtype="float32"
    )
    tf_res = tf_model(img, training=False).numpy()

    pt_img = torch.Tensor(img.transpose([0, 3, 1, 2]))
    pt_res = pt_model.forward(pt_img).detach().numpy()

    if model_name.startswith("deit_") and "distilled" in model_name:
        # During inference timm distilled models return average of both heads, while
        # we return both heads
        tf_res = tf.reduce_mean(tf_res, axis=1)

    # The tests are flaky sometimes, so we use a quite high tolerance
    assert (np.max(np.abs(tf_res - pt_res))) / (np.max(np.abs(pt_res)) + 1e-6) < 1e-3
