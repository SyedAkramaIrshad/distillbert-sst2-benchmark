from pathlib import Path

from mini_distill.metrics import folder_size_mb


def test_folder_size_mb(tmp_path: Path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"x" * 1024 * 1024)  # 1MB
    size = folder_size_mb(tmp_path)
    assert 0.95 <= size <= 1.05
