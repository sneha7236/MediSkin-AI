"""Dataset preparation: clean -> stratified split -> resize -> processed/ + manifest.

Expected raw layout (one folder per class, folder name = class slug):

    data/raw/<class_slug>/*.jpg|jpeg|png

Lesion-aware split: if file names look like ``<group>__<anything>.jpg`` (see
``training/organize_ham10000.py``), all images sharing a ``<group>`` (e.g. several photos of one
lesion) always land in the same split, so test metrics are not inflated by near-duplicates.
Without such prefixes every image is its own group (plain stratified split).

Output:
    data/processed/<split>/<class_slug>/*.png     (resized with the SAME function used at inference)
    data/metadata/dataset_manifest.csv            (split, label, path)

Usage:
    python -m training.preprocess [--raw-dir data/raw] [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from config import config, constants
from prediction.image_processor import resize_image
from utils.logger import get_logger

logger = get_logger("preprocess")

IMAGE_SUFFIXES = {f".{e}" for e in constants.ALLOWED_EXTENSIONS}
MIN_IMAGES_PER_CLASS = 6
SPLITS = ("train", "val", "test")
MANIFEST_NAME = "dataset_manifest.csv"


class DatasetError(RuntimeError):
    pass


class EmptyDatasetError(DatasetError):
    pass


@dataclass
class DatasetSummary:
    class_names: list[str]
    counts: dict[str, dict[str, int]]  # split -> class -> n
    removed: dict[str, int] = field(default_factory=dict)  # reason -> n
    manifest_path: Path | None = None


def _discover_classes(raw_dir: Path) -> dict[str, list[Path]]:
    if not raw_dir.is_dir():
        raise EmptyDatasetError(f"Dataset folder not found: {raw_dir}")
    classes: dict[str, list[Path]] = {}
    for folder in sorted(p for p in raw_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        files = sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)
        if files:
            classes[folder.name.strip().lower()] = files
    if not classes:
        raise EmptyDatasetError(
            f"No images found under {raw_dir}. Expected data/raw/<class_name>/<image>.jpg"
        )
    return classes


def _clean(classes: dict[str, list[Path]]) -> tuple[dict[str, list[Path]], dict[str, int]]:
    """Drop unreadable, tiny and duplicate (byte-identical) images."""
    removed = {"unreadable": 0, "too_small": 0, "duplicate": 0}
    seen_hashes: set[str] = set()
    cleaned: dict[str, list[Path]] = {}
    for label, files in classes.items():
        keep = []
        for path in files:
            try:
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    width, height = img.size
            except Exception:
                removed["unreadable"] += 1
                continue
            if min(width, height) < constants.MIN_IMAGE_SIDE:
                removed["too_small"] += 1
                continue
            if digest in seen_hashes:
                removed["duplicate"] += 1
                continue
            seen_hashes.add(digest)
            keep.append(path)
        cleaned[label] = keep
    return cleaned, removed


def _group_ids(paths: list[Path]) -> list[str]:
    """Group id from a ``<group>__<rest>`` file name; otherwise each image is its own group."""
    return [p.stem.split("__", 1)[0] if "__" in p.stem else f"_single_{i}" for i, p in enumerate(paths)]


def _grouped_fold(indices: list[int], labels: list[str], groups: list[str], fraction: float, seed: int):
    """Hold out roughly ``fraction`` of ``indices`` as whole groups, stratified by label."""
    n_splits = max(2, round(1.0 / fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest, held = next(splitter.split(indices, [labels[i] for i in indices], [groups[i] for i in indices]))
    return [indices[i] for i in rest], [indices[i] for i in held]


def _split(paths: list[Path], labels: list[str], val_size: float, test_size: float, seed: int) -> dict[str, list[int]]:
    indices = list(range(len(paths)))
    groups = _group_ids(paths)
    relative_val = val_size / (1.0 - test_size)
    if len(set(groups)) < len(paths):  # lesion-aware split
        train_val_idx, test_idx = _grouped_fold(indices, labels, groups, test_size, seed)
        train_idx, val_idx = _grouped_fold(train_val_idx, labels, groups, relative_val, seed)
        return {"train": train_idx, "val": val_idx, "test": test_idx}
    train_val_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=labels, random_state=seed)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val, stratify=[labels[i] for i in train_val_idx], random_state=seed
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def prepare_dataset(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    metadata_dir: Path | None = None,
    image_size: tuple[int, int] = constants.MODEL_INPUT_SIZE,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> DatasetSummary:
    raw_dir = Path(raw_dir or config.RAW_DATA_DIR)
    processed_dir = Path(processed_dir or config.PROCESSED_DATA_DIR)
    metadata_dir = Path(metadata_dir or config.METADATA_DIR)

    classes, removed = _clean(_discover_classes(raw_dir))
    classes = {k: v for k, v in classes.items() if v}
    if len(classes) < 2:
        raise DatasetError("At least two classes with valid images are required.")
    too_few = {k: len(v) for k, v in classes.items() if len(v) < MIN_IMAGES_PER_CLASS}
    if too_few:
        raise DatasetError(
            f"Each class needs at least {MIN_IMAGES_PER_CLASS} valid images. Too few: {too_few}"
        )

    paths = [p for files in classes.values() for p in files]
    labels = [label for label, files in classes.items() for _ in files]
    assignment = _split(paths, labels, val_size, test_size, seed)

    # Rebuild processed splits from scratch (only our own split folders are removed).
    for split in SPLITS:
        shutil.rmtree(processed_dir / split, ignore_errors=True)

    rows = []
    for split, idxs in assignment.items():
        for i in idxs:
            label = labels[i]
            out_dir = processed_dir / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{hashlib.md5(str(paths[i]).encode()).hexdigest()[:12]}.png"
            with Image.open(paths[i]) as img:
                resize_image(img, image_size).save(out_path, format="PNG")
            rows.append({"split": split, "label": label, "path": str(out_path.relative_to(processed_dir))})

    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows).sort_values(["split", "label", "path"]).reset_index(drop=True)
    manifest_path = metadata_dir / MANIFEST_NAME
    manifest.to_csv(manifest_path, index=False)

    counts = {
        split: manifest[manifest.split == split].label.value_counts().sort_index().to_dict()
        for split in SPLITS
    }
    summary = DatasetSummary(sorted(classes), counts, removed, manifest_path)
    logger.info("Dataset prepared: %s classes, removed=%s", len(classes), removed)
    return summary


def load_manifest(metadata_dir: Path | None = None, processed_dir: Path | None = None) -> pd.DataFrame:
    """Manifest with an absolute ``filepath`` column."""
    metadata_dir = Path(metadata_dir or config.METADATA_DIR)
    processed_dir = Path(processed_dir or config.PROCESSED_DATA_DIR)
    path = metadata_dir / MANIFEST_NAME
    if not path.is_file():
        raise EmptyDatasetError("Dataset manifest not found. Run `python -m training.preprocess` first.")
    manifest = pd.read_csv(path)
    if manifest.empty:
        raise EmptyDatasetError("Dataset manifest is empty.")
    manifest["filepath"] = manifest["path"].map(lambda p: str(processed_dir / p))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, split and resize the raw dataset.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()
    try:
        summary = prepare_dataset(args.raw_dir, val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    except DatasetError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(f"Classes ({len(summary.class_names)}): {summary.class_names}")
    for split, per_class in summary.counts.items():
        print(f"  {split}: {sum(per_class.values())} images  {per_class}")
    print(f"Removed during cleaning: {summary.removed}")
    print(f"Manifest: {summary.manifest_path}")


if __name__ == "__main__":
    main()
