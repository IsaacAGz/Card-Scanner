import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from image_crops import extract_images_from_zip, load_image_files, process_images_crops, process_images_crops_from_uploads


def _jpeg_bytes() -> bytes:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[40:200, 80:240] = 255
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


class FakeYolo:
    def __call__(self, frame, save=False, conf=0.75):
        class Box:
            xyxy = [np.array([80.0, 40.0, 240.0, 200.0])]

        class Result:
            boxes = [Box()] if frame.sum() > 0 else None

        class Results:
            def __getitem__(self, index):
                return [Result()][index]

        return Results()


def test_extract_valid_zip_with_mixed_files(tmp_path: Path):
    jpeg = _jpeg_bytes()
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in ["a.jpg", "b.jpg", "nested/c.jpg"]:
            zf.writestr(name, jpeg)
        zf.writestr("notes.txt", b"ignore me")

    extracted = extract_images_from_zip(archive.read_bytes(), max_images=10, max_uncompressed_bytes=10_000_000)
    assert len(extracted) == 3
    assert all(name.endswith(".jpg") for name, _ in extracted)


@pytest.mark.parametrize(
    "bad_name",
    ["../../etc/passwd.jpg", "../passwd.jpg", "/etc/passwd.jpg", "nested/../../passwd.jpg"],
)
def test_reject_zip_slip_paths(tmp_path: Path, bad_name: str):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(bad_name, b"fake")

    with pytest.raises(ValueError, match="Unsafe path"):
        extract_images_from_zip(archive.read_bytes(), max_images=10, max_uncompressed_bytes=10_000_000)


def test_reject_oversized_uncompressed_total(tmp_path: Path):
    jpeg = _jpeg_bytes()
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for index in range(20):
            zf.writestr(f"img{index}.jpg", jpeg)

    with pytest.raises(ValueError, match="maximum uncompressed size"):
        extract_images_from_zip(archive.read_bytes(), max_images=50, max_uncompressed_bytes=500)


def test_max_images_cap(tmp_path: Path):
    jpeg = _jpeg_bytes()
    archive = tmp_path / "many.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for index in range(5):
            zf.writestr(f"img{index}.jpg", jpeg)

    with pytest.raises(ValueError, match="more than 3 images"):
        extract_images_from_zip(archive.read_bytes(), max_images=3, max_uncompressed_bytes=10_000_000)


def test_skip_macosx_and_dotfiles(tmp_path: Path):
    archive = tmp_path / "meta.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("__MACOSX/._a.jpg", b"junk")
        zf.writestr(".hidden.jpg", b"junk")
        zf.writestr("real.jpg", _jpeg_bytes())

    extracted = extract_images_from_zip(archive.read_bytes(), max_images=10, max_uncompressed_bytes=10_000_000)
    assert extracted == [("real.jpg", extracted[0][1])]
    assert len(extracted) == 1


def test_zip_pipeline_matches_direct_files(tmp_path: Path):
    jpeg = _jpeg_bytes()
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in ["a.jpg", "b.jpg", "nested/c.jpg"]:
            zf.writestr(name, jpeg)

    from_zip = process_images_crops_from_uploads(zip_file=archive.read_bytes(), yolo=FakeYolo())
    assert from_zip.crop_count == 3

    paths = []
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        path = tmp_path / name
        path.write_bytes(jpeg)
        paths.append(path)

    from_files = process_images_crops(load_image_files(paths), yolo=FakeYolo())
    assert from_files.crop_count == 3


def test_mutually_exclusive_inputs():
    with pytest.raises(ValueError, match="either image_files or zip_file"):
        process_images_crops_from_uploads(
            image_files=[("a.jpg", b"x")],
            zip_file=b"PK",
            yolo=FakeYolo(),
        )
