import csv

import pytest

from unityvideo.data import PairedVideoDataset


def write_metadata(path, columns):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: column for column in columns})


def test_metadata_contract(tmp_path):
    metadata = tmp_path / "metadata.csv"
    write_metadata(metadata, ["video", "prompt", "depth"])
    dataset = PairedVideoDataset(metadata, "depth", 33, 256, 256)
    assert len(dataset) == 1
    assert dataset._resolve("clip.mp4") == tmp_path / "clip.mp4"


def test_metadata_requires_modality_column(tmp_path):
    metadata = tmp_path / "metadata.csv"
    write_metadata(metadata, ["video", "prompt"])
    with pytest.raises(ValueError, match="depth"):
        PairedVideoDataset(metadata, "depth", 33, 256, 256)
