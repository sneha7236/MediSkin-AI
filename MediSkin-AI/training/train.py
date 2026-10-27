"""Train the skin-condition classifier.

Pipeline: prepared dataset -> augmentation (train only) -> CNN / transfer learning with
BatchNormalization -> Adam -> validation -> test evaluation -> save model + metrics.

Usage:
    python -m training.preprocess                       # once, after placing data in data/raw
    python -m training.train --arch mobilenetv2 --epochs 15 --fine-tune-epochs 10
    python -m training.train --arch cnn --epochs 30     # no pretrained-weight download needed

Model input contract: float32 RGB in [0, 1]. Rescaling for the backbone is part of the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from config import config, constants
from training.augmentation import build_augmentation
from training.evaluate import compute_metrics, predict_dataset, save_metrics
from training.preprocess import DatasetError, EmptyDatasetError, load_manifest
from utils.logger import get_logger

logger = get_logger("train")

ARCHITECTURES = ("mobilenetv2", "cnn")


def read_class_names(path: Path | None = None) -> list[str]:
    path = Path(path or config.CLASS_NAMES_PATH)
    if not path.is_file():
        raise EmptyDatasetError("models/class_names.json not found. Train a model first.")
    return json.loads(path.read_text(encoding="utf-8"))


def make_dataset(df, class_names: list[str], batch_size: int, image_size: tuple[int, int],
                 training: bool, seed: int = 42):
    """tf.data pipeline over processed PNGs. Labels are integer indices into class_names."""
    import tensorflow as tf

    index = {name: i for i, name in enumerate(class_names)}
    paths = df["filepath"].tolist()
    labels = np.array([index[l] for l in df["label"]], dtype=np.int32)
    width, height = image_size

    def load(path, label):
        data = tf.io.read_file(path)
        image = tf.io.decode_png(data, channels=3)
        image = tf.ensure_shape(image, [None, None, 3])
        image = tf.image.resize(image, (height, width))  # no-op for prepared images
        return tf.cast(image, tf.float32) / 255.0, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if training:
        augment = build_augmentation()
        ds = ds.map(lambda x, y: (tf.clip_by_value(augment(x, training=True), 0.0, 1.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def _head(x, num_classes: int):
    from keras import layers

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(0.3)(x)
    return layers.Dense(num_classes, activation="softmax", name="predictions")(x)


def build_model(num_classes: int, arch: str = "mobilenetv2",
                image_size: tuple[int, int] = constants.MODEL_INPUT_SIZE, pretrained: bool = True):
    """Return (model, backbone_or_None, pretrained_actually_used)."""
    import keras
    from keras import layers

    width, height = image_size
    inputs = keras.Input(shape=(height, width, 3), name="image")

    if arch == "mobilenetv2":
        weights = "imagenet" if pretrained else None
        try:
            backbone = keras.applications.MobileNetV2(
                input_shape=(height, width, 3), include_top=False, weights=weights)
        except Exception as exc:  # e.g. offline: weights cannot be downloaded
            raise DatasetError(
                f"Could not load MobileNetV2 ImageNet weights ({type(exc).__name__}). "
                "Check your internet connection, or train with --no-pretrained / --arch cnn."
            ) from exc
        backbone.trainable = False
        x = layers.Rescaling(scale=2.0, offset=-1.0, name="to_minus1_1")(inputs)  # [0,1] -> [-1,1]
        x = backbone(x, training=False)  # keeps backbone BatchNorm in inference mode
        model = keras.Model(inputs, _head(x, num_classes), name="mediskin_mobilenetv2")
        return model, backbone, weights is not None

    if arch == "cnn":
        x = inputs
        for i, filters in enumerate((32, 64, 128, 128), start=1):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
            x = layers.BatchNormalization(name=f"bn{i}")(x)
            x = layers.Activation("relu", name=("gradcam_target" if i == 4 else f"relu{i}"))(x)
            if i < 4:
                x = layers.MaxPooling2D(name=f"pool{i}")(x)
        model = keras.Model(inputs, _head(x, num_classes), name="mediskin_cnn")
        return model, None, False

    raise ValueError(f"Unknown architecture '{arch}'. Choose from {ARCHITECTURES}.")


def _compile(model, learning_rate: float):
    import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def _callbacks(patience: int):
    import keras

    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(1, patience // 2)),
    ]


def train(arch: str, epochs: int, fine_tune_epochs: int, batch_size: int, learning_rate: float,
          pretrained: bool, image_size: tuple[int, int], seed: int = 42) -> dict:
    import keras

    keras.utils.set_random_seed(seed)
    manifest = load_manifest()
    class_names = sorted(manifest["label"].unique())
    train_df, val_df, test_df = (manifest[manifest.split == s] for s in ("train", "val", "test"))
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise EmptyDatasetError("Train/validation/test split is empty. Re-run preprocessing with more data.")

    train_ds = make_dataset(train_df, class_names, batch_size, image_size, training=True, seed=seed)
    val_ds = make_dataset(val_df, class_names, batch_size, image_size, training=False)
    test_ds = make_dataset(test_df, class_names, batch_size, image_size, training=False)

    y_train = np.array([class_names.index(l) for l in train_df["label"]])
    weights = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    model, backbone, used_pretrained = build_model(len(class_names), arch, image_size, pretrained)
    if arch == "mobilenetv2" and not used_pretrained:
        logger.warning("Training MobileNetV2 from random initialisation (no pretrained weights).")

    _compile(model, learning_rate)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=class_weight, callbacks=_callbacks(patience=5), verbose=2)
    histories = {"stage1": history.history}

    if backbone is not None and fine_tune_epochs > 0:
        backbone.trainable = True
        for layer in backbone.layers[:-40]:  # keep early layers frozen
            layer.trainable = False
        for layer in backbone.layers:
            if layer.__class__.__name__ == "BatchNormalization":
                layer.trainable = False  # keep BN statistics stable while fine-tuning
        _compile(model, learning_rate / 10)
        fine = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs,
                         class_weight=class_weight, callbacks=_callbacks(patience=4), verbose=2)
        histories["fine_tune"] = fine.history

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.MODEL_PATH)
    config.CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    config.HISTORY_PATH.write_text(
        json.dumps({s: {k: [float(v) for v in vals] for k, vals in h.items()} for s, h in histories.items()}, indent=2),
        encoding="utf-8")

    y_true, y_pred = predict_dataset(model, test_ds)
    metrics = compute_metrics(y_true, y_pred, class_names)
    save_metrics(metrics, extra={
        "model": {"architecture": arch, "pretrained_weights": used_pretrained,
                  "input_size": list(image_size), "epochs_requested": epochs,
                  "fine_tune_epochs_requested": fine_tune_epochs if backbone is not None else 0},
        "dataset": {s: int(len(d)) for s, d in (("train", train_df), ("val", val_df), ("test", test_df))},
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MediSkin AI classifier.")
    parser.add_argument("--arch", choices=ARCHITECTURES, default="mobilenetv2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=constants.MODEL_INPUT_SIZE[0])
    parser.add_argument("--no-pretrained", action="store_true", help="Do not use ImageNet weights")
    args = parser.parse_args()
    try:
        metrics = train(args.arch, args.epochs, args.fine_tune_epochs, args.batch_size,
                        args.learning_rate, not args.no_pretrained, (args.image_size, args.image_size))
    except DatasetError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(f"\nTest accuracy: {metrics['accuracy']:.4f} | macro F1: {metrics['macro']['f1']:.4f} "
          f"(n={metrics['num_test_samples']})")
    print(f"Saved: {config.MODEL_PATH}, {config.CLASS_NAMES_PATH}, {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
