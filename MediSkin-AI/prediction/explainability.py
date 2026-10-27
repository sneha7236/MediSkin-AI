"""Grad-CAM for linear-chain Keras classifiers (Input -> layers -> softmax).

The target feature map is either a layer named ``gradcam_target`` (custom CNN) or the output
of a nested backbone model such as MobileNetV2 (its un-pooled last convolutional features).
If the model does not fit this structure, ``ExplainabilityUnavailable`` is raised and the UI
simply hides the section; nothing is faked.
"""
from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageOps

from prediction.image_processor import resize_image

TARGET_LAYER_NAME = "gradcam_target"


class ExplainabilityUnavailable(RuntimeError):
    pass


def _chain_layers(model):
    import keras

    layers = [l for l in model.layers if not isinstance(l, keras.layers.InputLayer)]
    if not layers:
        raise ExplainabilityUnavailable("Model has no layers.")
    # A linear chain means each layer consumes the previous layer's output.
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ExplainabilityUnavailable("Model must have one input and one output.")
    return layers


def _find_target_index(layers) -> int:
    import keras

    for i, layer in enumerate(layers):
        if layer.name == TARGET_LAYER_NAME:
            return i
    for i in range(len(layers) - 1, -1, -1):
        layer = layers[i]
        shape = getattr(layer, "output", None)
        shape = getattr(shape, "shape", None)
        if isinstance(layer, keras.Model) and shape is not None and len(shape) == 4:
            return i
    raise ExplainabilityUnavailable("No convolutional feature map found.")


def compute_gradcam(model, batch: np.ndarray, class_index: int | None = None) -> np.ndarray:
    """Return a heatmap in [0, 1] of shape (h, w) for the predicted (or given) class."""
    import tensorflow as tf

    layers = _chain_layers(model)
    target_index = _find_target_index(layers)

    x = tf.convert_to_tensor(batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_output = None
        for i, layer in enumerate(layers):
            x = layer(x, training=False)
            if i == target_index:
                conv_output = x
                tape.watch(conv_output)
        predictions = x
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        raise ExplainabilityUnavailable("Gradients could not be computed.")
    weights = tf.reduce_mean(grads, axis=(1, 2))  # (1, C)
    cam = tf.reduce_sum(conv_output * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    peak = cam.max()
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam / peak).astype(np.float32)


def resize_for_display(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return resize_image(image, size)


def render_heatmap(base: Image.Image, heatmap: np.ndarray, alpha: float = 0.45):
    """Return (original, heatmap_image, overlay_image) as RGB PIL images the size of ``base``."""
    heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(base.size, Image.Resampling.BICUBIC)
    heat_arr = np.asarray(heat, dtype=np.float32) / 255.0
    colored = (colormaps["jet"](heat_arr)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(colored, "RGB")
    overlay = Image.blend(base.convert("RGB"), heat_img, alpha)
    return base.convert("RGB"), heat_img, overlay
