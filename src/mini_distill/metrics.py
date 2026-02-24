from __future__ import annotations

from pathlib import Path


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def param_size_mb_fp32(model) -> float:
    # fp32 ~= 4 bytes/parameter
    return count_parameters(model) * 4 / (1024 ** 2)


def folder_size_mb(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 ** 2)
