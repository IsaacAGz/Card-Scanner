from pathlib import Path

from scripts.check_artifacts import REPO_ROOT, resolve_path


def test_resolve_path_ok_when_runtime_exists(tmp_path: Path):
    runtime = tmp_path / "weights.pt"
    runtime.write_bytes(b"x")
    path, status = resolve_path(runtime, tmp_path / "missing.pt")
    assert path == runtime
    assert status == "ok"


def test_resolve_path_missing_when_neither_exists():
    runtime = REPO_ROOT / "tests" / "missing-runtime.bin"
    _, status = resolve_path(runtime, REPO_ROOT / "tests" / "missing-fallback.bin")
    assert status == "missing"


def test_resolve_path_reports_fallback():
    runtime = REPO_ROOT / "tests" / "missing-runtime.bin"
    fallback = REPO_ROOT / "tests" / "_fallback_probe.db"
    fallback.write_bytes(b"db")
    try:
        _, status = resolve_path(runtime, fallback)
    finally:
        fallback.unlink(missing_ok=True)
    assert status.startswith("missing at tests")
    assert "found at tests" in status
