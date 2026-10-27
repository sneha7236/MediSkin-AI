"""Arrange the HAM10000 dataset into the layout ``training.preprocess`` expects.

HAM10000 ships as image folders plus ``HAM10000_metadata.csv`` (columns ``image_id``, ``lesion_id``,
``dx``). This copies every image to ``<dest>/<dx>/<lesion_id>__<image_id>.jpg`` so that the
lesion-aware split in ``training.preprocess`` keeps all photos of one lesion in the same split.

Usage:
    python -m training.organize_ham10000 --source path/to/HAM10000 [--dest data/raw]

The dataset itself is NOT included in this project; download it yourself (see README) and respect
its licence (CC BY-NC 4.0 for HAM10000).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from config import config
from training.preprocess import IMAGE_SUFFIXES, DatasetError, EmptyDatasetError

REQUIRED_COLUMNS = {"image_id", "lesion_id", "dx"}


def find_metadata(source: Path) -> Path:
    matches = sorted(source.rglob("*metadata*.csv"))
    if not matches:
        raise EmptyDatasetError(f"No HAM10000 metadata CSV found under {source}.")
    return matches[0]


def organize(source: Path, dest: Path | None = None) -> dict[str, int]:
    """Copy images into class folders. Returns {class: count}; raises DatasetError on bad input."""
    source, dest = Path(source), Path(dest or config.RAW_DATA_DIR)
    if not source.is_dir():
        raise EmptyDatasetError(f"Source folder not found: {source}")
    metadata = pd.read_csv(find_metadata(source))
    if not REQUIRED_COLUMNS <= set(metadata.columns):
        raise DatasetError(f"Metadata CSV must contain the columns {sorted(REQUIRED_COLUMNS)}.")

    images = {p.stem: p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    counts: dict[str, int] = {}
    missing = 0
    for row in metadata.itertuples(index=False):
        image = images.get(str(row.image_id))
        label = str(row.dx).strip().lower()
        if image is None or not label:
            missing += 1
            continue
        target_dir = dest / label
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target_dir / f"{row.lesion_id}__{row.image_id}{image.suffix.lower()}")
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        raise EmptyDatasetError("None of the images listed in the metadata were found in the source folder.")
    if missing:
        print(f"Warning: {missing} metadata rows had no matching image and were skipped.")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Arrange HAM10000 into data/raw/<class>/ folders.")
    parser.add_argument("--source", type=Path, required=True, help="Folder containing the images and metadata CSV.")
    parser.add_argument("--dest", type=Path, default=config.RAW_DATA_DIR)
    args = parser.parse_args()
    try:
        counts = organize(args.source, args.dest)
    except DatasetError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(f"Copied {sum(counts.values())} images into {args.dest}: {dict(sorted(counts.items()))}")
    print("Next: python -m training.preprocess")


if __name__ == "__main__":
    main()
