#!/usr/bin/env python3
"""Copy Kaiwu-allowed files into a local upload batch directory."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from sync_manifest import DEFAULT_ALLOWED_PATHS, ROOT, build_manifest, normalize_rel


DEFAULT_OUTPUT = Path("automation/data/sync_batch")


def git_changed_files() -> set[Path]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-uall"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed: set[Path] = set()
    for line in result.stdout.splitlines():
        if not line:
            continue
        raw_path = line[3:]
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        raw_path = raw_path.strip().strip('"')
        if raw_path:
            changed.add(Path(raw_path))
    return changed


def is_under(path: Path, root: Path) -> bool:
    if path == root:
        return True
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def filter_paths(files: Iterable[Path], allowed_paths: Sequence[Path]) -> list[Path]:
    allowed = []
    for file_path in files:
        if any(is_under(file_path, allowed_path) for allowed_path in allowed_paths):
            absolute = ROOT / file_path
            if absolute.is_file():
                allowed.append(file_path)
    return sorted(set(allowed), key=lambda item: item.as_posix())


def strip_previous_stamp(lines: list[str]) -> list[str]:
    return [line for line in lines if not line.startswith("# Uploaded at: ")]


def inject_upload_stamp(text: str, stamp: str) -> str:
    lines = strip_previous_stamp(text.splitlines(keepends=True))
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    if len(lines) > insert_at and "coding" in lines[insert_at]:
        insert_at += 1
    lines.insert(insert_at, f"# Uploaded at: {stamp}\n")
    return "".join(lines)


def reset_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def copy_batch(files: Sequence[Path], output: Path, dry_run: bool = False) -> dict:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    copied = []
    if not dry_run:
        reset_output(output)
    for rel in files:
        src = ROOT / rel
        dst = output / rel
        copied.append(normalize_rel(rel))
        if dry_run:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".py":
            text = src.read_text(encoding="utf-8")
            dst.write_text(inject_upload_stamp(text, stamp), encoding="utf-8", newline="\n")
        else:
            shutil.copy2(src, dst)
    return {
        "generated_at": stamp,
        "output": str(output),
        "file_count": len(copied),
        "files": copied,
        "dry_run": dry_run,
    }


def resolve_files(scope: str, paths: Sequence[Path]) -> list[Path]:
    if scope == "changed":
        return filter_paths(git_changed_files(), paths)
    manifest = build_manifest(paths)
    return [Path(item["path"]) for item in manifest["files"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=["full", "changed"],
        default="full",
        help="Build a full allowed batch or only currently changed files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Batch output directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the batch plan without copying files.")
    args = parser.parse_args()

    output = ROOT / args.output
    files = resolve_files(args.scope, DEFAULT_ALLOWED_PATHS)
    manifest = copy_batch(files, output, dry_run=args.dry_run)
    manifest_path = output.parent / f"{output.name}_manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{'Would copy' if args.dry_run else 'Copied'} {len(files)} files to {output}")


if __name__ == "__main__":
    main()
