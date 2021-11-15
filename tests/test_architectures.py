import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from tfimm import create_model


# We test one (small) model from each architecture. Even then tests take time
@pytest.mark.parametrize("model_name", ["resnet18"]) # , "vit_tiny_patch16_224"])
@pytest.mark.timeout(60)
def test_save_load_model_h5(model_name):
    model = create_model(model_name)

    # Save model and load it again
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model"
        model.save(model_path)
        loaded_model = tf.keras.models.load_model(model_path)

    # Run inference on original and loaded models
    img = model.dummy_inputs
    res = model(img).numpy()
    res_2 = loaded_model(img).numpy()
    assert np.allclose(res, res_2)