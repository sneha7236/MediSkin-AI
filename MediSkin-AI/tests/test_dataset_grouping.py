"""Lesion-aware dataset split and the HAM10000 organiser."""
import pandas as pd
import pytest
from PIL import Image

from training import organize_ham10000 as organizer
from training.preprocess import DatasetError, EmptyDatasetError, load_manifest, prepare_dataset


def make_image(path, shade):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (shade % 256, (shade * 3) % 256, (shade * 7) % 256)).save(path)


def build_ham_source(tmp_path, lesions_per_class=12, photos_per_lesion=3):
    source = tmp_path / "ham"
    rows, shade = [], 1
    for dx in ("mel", "nv", "bcc"):
        for lesion in range(lesions_per_class):
            for photo in range(photos_per_lesion):
                image_id = f"ISIC_{dx}_{lesion}_{photo}"
                make_image(source / "HAM10000_images_part_1" / f"{image_id}.jpg", shade)
                rows.append({"image_id": image_id, "lesion_id": f"HAM_{dx}_{lesion}", "dx": dx.upper()})
                shade += 1
    pd.DataFrame(rows).to_csv(source / "HAM10000_metadata.csv", index=False)
    return source


def test_organizer_copies_into_class_folders_with_lesion_prefix(tmp_path):
    source = build_ham_source(tmp_path)
    counts = organizer.organize(source, tmp_path / "raw")
    assert counts == {"mel": 36, "nv": 36, "bcc": 36}
    sample = next((tmp_path / "raw" / "mel").iterdir())
    assert sample.name.startswith("HAM_mel_") and "__ISIC_mel_" in sample.name


def test_organizer_rejects_bad_input(tmp_path):
    with pytest.raises(EmptyDatasetError):
        organizer.organize(tmp_path / "missing", tmp_path / "raw")
    (tmp_path / "empty").mkdir()
    with pytest.raises(EmptyDatasetError):
        organizer.organize(tmp_path / "empty", tmp_path / "raw")
    bad = tmp_path / "bad"
    bad.mkdir()
    pd.DataFrame({"foo": [1]}).to_csv(bad / "metadata.csv", index=False)
    with pytest.raises(DatasetError):
        organizer.organize(bad, tmp_path / "raw")
    pd.DataFrame({"image_id": ["x"], "lesion_id": ["l"], "dx": ["mel"]}).to_csv(bad / "metadata.csv", index=False)
    with pytest.raises(EmptyDatasetError):  # metadata lists images that are not present
        organizer.organize(bad, tmp_path / "raw")


def test_organized_dataset_prepares_all_three_splits(tmp_path):
    organizer.organize(build_ham_source(tmp_path), tmp_path / "raw")
    prepare_dataset(tmp_path / "raw", tmp_path / "proc", tmp_path / "meta", image_size=(32, 32))
    manifest = load_manifest(tmp_path / "meta", tmp_path / "proc")
    assert set(manifest.split) == {"train", "val", "test"}
    assert set(manifest.label) == {"mel", "nv", "bcc"}
    assert len(manifest) == 108


def test_grouped_split_directly_keeps_groups_together(tmp_path):
    from pathlib import Path

    from training.preprocess import _group_ids, _split

    paths = [Path(f"g{g}__img{i}.jpg") for g in range(30) for i in range(3)]
    labels = ["a" if g % 2 else "b" for g in range(30) for _ in range(3)]
    assignment = _split(paths, labels, 0.15, 0.15, seed=1)
    groups = _group_ids(paths)
    owner = {}
    for split, idxs in assignment.items():
        for i in idxs:
            assert owner.setdefault(groups[i], split) == split, "group split across partitions"
    assert sum(len(v) for v in assignment.values()) == len(paths)
    assert all(assignment[s] for s in ("train", "val", "test"))


def test_ungrouped_files_use_plain_stratified_split():
    from pathlib import Path

    from training.preprocess import _group_ids

    ids = _group_ids([Path("a.jpg"), Path("b.jpg"), Path("c.jpg")])
    assert len(set(ids)) == 3
