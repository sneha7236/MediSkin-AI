"""Training-time image augmentation (applied only to the training split)."""
from __future__ import annotations


def build_augmentation():
    """Keras preprocessing layers; call with ``training=True`` on a batch in [0, 1].

    Conservative geometric and photometric changes appropriate for skin photographs.
    """
    import keras
    from keras import layers

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15, fill_mode="reflect"),
            layers.RandomZoom(0.15, fill_mode="reflect"),
            layers.RandomTranslation(0.08, 0.08, fill_mode="reflect"),
            layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )
