#!/usr/bin/env python3
"""Build a manifest for files that may be synchronized to Tencent Kaiwu."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWED_PATHS = (
    Path("agent_diy"),
    Path("agent_ppo"),
    Path("conf"),
    Path("log"),
    Path("train_test.py"),
)
SKIP_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log", ".zip"}


def normalize_rel(path: Path) -> str:
    return path.as_posix()


def is_skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts) or path.suffix in SKIP_SUFFIXES


def iter_files(paths: Sequence[Path]) -> Iterable[Path]:
    for rel in paths:
        absolute = ROOT / rel
        if not absolute.exists():
            continue
        if absolute.is_file():
            if not is_skipped(rel):
                yield rel
            continue
        for file_path in sorted(absolute.rglob("*")):
            if not file_path.is_file():
                continue
            file_rel = file_path.relative_to(ROOT)
            if not is_skipped(file_rel):
                yield file_rel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(paths: Sequence[Path]) -> dict:
    files = []
    for rel in sorted(set(iter_files(paths)), key=lambda item: item.as_posix()):
        absolute = ROOT / rel
        files.append(
            {
                "path": normalize_rel(rel),
                "size": absolute.stat().st_size,
                "sha256": sha256_file(absolute),
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "allowed_paths": [normalize_rel(path) for path in paths],
        "file_count": len(files),
        "files": files,
    }


def parse_paths(values: Sequence[str] | None) -> list[Path]:
    if not values:
        return list(DEFAULT_ALLOWED_PATHS)
    return [Path(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        help="Allowed relative path to include. May be repeated.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("automation/data/sync_manifest.json"),
        help="Output JSON manifest path.",
    )
    args = parser.parse_args()

    manifest = build_manifest(parse_paths(args.paths))
    output = ROOT / args.json_out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest with {manifest['file_count']} files: {output}")


if __name__ == "__main__":
    main()
