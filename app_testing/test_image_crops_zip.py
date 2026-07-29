"""Phase 3 acceptance tests for ZIP-as-input image crop extraction."""

from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"

sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

from image_crops import (  # noqa: E402
    extract_images_from_zip,
    load_image_files,
    process_images_crops,
    process_images_crops_from_uploads,
)


def make_card_jpeg(path: Path) -> bytes:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[40:200, 80:240] = 255
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    data = encoded.tobytes()
    path.write_bytes(data)
    return data


class FakeYolo:
    def __call__(self, frame, save=False, conf=0.75):
        class Box:
            xyxy = [np.array([80.0, 40.0, 240.0, 200.0])]

        class Result:
            boxes = [Box()] if frame.sum() > 0 else None

        class Results:
            def __getitem__(self, i):
                return [Result()][i]

            def __len__(self):
                return 1

        return Results()


def test_extract_valid_zip_with_mixed_files(tmpdir: Path) -> None:
    input_zip = tmpdir / "input.zip"
    jpeg = make_card_jpeg(tmpdir / "x.jpg")
    with zipfile.ZipFile(input_zip, "w") as zf:
        for name in ["a.jpg", "b.jpg", "nested/c.jpg"]:
            zf.writestr(name, jpeg)
        zf.writestr("notes.txt", b"ignore me")

    extracted = extract_images_from_zip(
        input_zip.read_bytes(),
        max_images=10,
        max_uncompressed_bytes=10_000_000,
    )
    assert len(extracted) == 3
    assert all(name.endswith(".jpg") for name, _ in extracted)


def test_reject_zip_slip_paths(tmpdir: Path) -> None:
    bad_names = [
        "../../etc/passwd.jpg",
        "../passwd.jpg",
        "/etc/passwd.jpg",
        "nested/../../passwd.jpg",
    ]
    for bad_name in bad_names:
        bad_zip = tmpdir / f"bad_{bad_name.replace('/', '_')}.zip"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr(bad_name, b"fake")
        try:
            extract_images_from_zip(
                bad_zip.read_bytes(),
                max_images=10,
                max_uncompressed_bytes=10_000_000,
            )
            raise AssertionError(f"expected zip slip failure for {bad_name!r}")
        except ValueError as exc:
            assert "Unsafe path" in str(exc), (bad_name, str(exc))


def test_reject_oversized_uncompressed_total(tmpdir: Path) -> None:
    bomb_zip = tmpdir / "bomb.zip"
    jpeg = make_card_jpeg(tmpdir / "x.jpg")
    with zipfile.ZipFile(bomb_zip, "w") as zf:
        for i in range(20):
            zf.writestr(f"img{i}.jpg", jpeg)

    try:
        extract_images_from_zip(
            bomb_zip.read_bytes(),
            max_images=50,
            max_uncompressed_bytes=500,
        )
        raise AssertionError("expected zip bomb failure")
    except ValueError as exc:
        assert "maximum uncompressed size" in str(exc)


def test_max_images_cap(tmpdir: Path) -> None:
    many_zip = tmpdir / "many.zip"
    jpeg = make_card_jpeg(tmpdir / "x.jpg")
    with zipfile.ZipFile(many_zip, "w") as zf:
        for i in range(5):
            zf.writestr(f"img{i}.jpg", jpeg)

    try:
        extract_images_from_zip(
            many_zip.read_bytes(),
            max_images=3,
            max_uncompressed_bytes=10_000_000,
        )
        raise AssertionError("expected max images failure")
    except ValueError as exc:
        assert "more than 3 images" in str(exc)


def test_skip_macosx_and_dotfiles(tmpdir: Path) -> None:
    meta_zip = tmpdir / "meta.zip"
    jpeg = make_card_jpeg(tmpdir / "x.jpg")
    with zipfile.ZipFile(meta_zip, "w") as zf:
        zf.writestr("__MACOSX/._a.jpg", b"junk")
        zf.writestr(".hidden.jpg", b"junk")
        zf.writestr("real.jpg", jpeg)

    extracted = extract_images_from_zip(
        meta_zip.read_bytes(),
        max_images=10,
        max_uncompressed_bytes=10_000_000,
    )
    assert len(extracted) == 1
    assert extracted[0][0] == "real.jpg"


def test_zip_pipeline_matches_direct_files(tmpdir: Path) -> None:
    input_zip = tmpdir / "input.zip"
    with zipfile.ZipFile(input_zip, "w") as zf:
        for name in ["a.jpg", "b.jpg", "nested/c.jpg"]:
            zf.writestr(name, make_card_jpeg(tmpdir / "x.jpg"))

    result_zip = process_images_crops_from_uploads(
        zip_file=input_zip.read_bytes(),
        yolo=FakeYolo(),
    )
    assert result_zip.crop_count == 3

    files = []
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        path = tmpdir / name
        make_card_jpeg(path)
        files.append(path)

    result_files = process_images_crops(load_image_files(files), yolo=FakeYolo())
    assert result_files.crop_count == 3


def test_mutually_exclusive_inputs() -> None:
    try:
        process_images_crops_from_uploads(
            image_files=[("a.jpg", b"x")],
            zip_file=b"PK",
            yolo=FakeYolo(),
        )
        raise AssertionError("expected mutual exclusion error")
    except ValueError as exc:
        assert "either image_files or zip_file" in str(exc)


def main() -> int:
    tmpdir = Path(tempfile.mkdtemp())
    test_extract_valid_zip_with_mixed_files(tmpdir)
    test_reject_zip_slip_paths(tmpdir)
    test_reject_oversized_uncompressed_total(tmpdir)
    test_max_images_cap(tmpdir)
    test_skip_macosx_and_dotfiles(tmpdir)
    test_zip_pipeline_matches_direct_files(tmpdir)
    test_mutually_exclusive_inputs()
    print("Phase 3 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
