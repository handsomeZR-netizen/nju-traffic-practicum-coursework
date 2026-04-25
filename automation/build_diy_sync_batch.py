#!/usr/bin/env python3
"""Build a focused upload batch for the Tencent Kaiwu DIY slot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from build_sync_batch import ROOT, copy_batch, filter_paths, git_changed_files
from sync_manifest import build_manifest


DIY_PATHS = (Path("agent_diy"), Path("train_test.py"))
CONF_PATHS = (Path("conf"),)
ALL_ALLOWED_PATHS = (Path("agent_diy"), Path("agent_ppo"), Path("conf"), Path("log"), Path("train_test.py"))
DEFAULT_OUTPUT = Path("automation/data/sync_batch_diy")


def selected_allowed_paths(args: argparse.Namespace) -> Sequence[Path]:
    if args.all_allowed:
        return ALL_ALLOWED_PATHS
    if args.with_conf:
        return DIY_PATHS + CONF_PATHS
    return DIY_PATHS


def resolve_files(args: argparse.Namespace, paths: Sequence[Path]) -> list[Path]:
    if args.full:
        manifest = build_manifest(paths)
        return [Path(item["path"]) for item in manifest["files"]]
    return filter_paths(git_changed_files(), paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-conf", action="store_true", help="Also include changed conf/** files.")
    parser.add_argument("--all-allowed", action="store_true", help="Include changed files from all Kaiwu-allowed paths.")
    parser.add_argument("--full", action="store_true", help="Include every file from the selected allowed paths.")
    parser.add_argument("--dry-run", action="store_true", help="Print the batch plan without copying files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Batch output directory.")
    args = parser.parse_args()

    paths = selected_allowed_paths(args)
    files = resolve_files(args, paths)
    output = ROOT / args.output
    manifest = copy_batch(files, output, dry_run=args.dry_run)
    manifest["selected_paths"] = [path.as_posix() for path in paths]
    manifest_path = output.parent / f"{output.name}_manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{'Would copy' if args.dry_run else 'Copied'} {len(files)} DIY batch files to {output}")


if __name__ == "__main__":
    main()
