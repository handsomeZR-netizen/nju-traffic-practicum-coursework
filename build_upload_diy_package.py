#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Build a minimal standalone upload package for the strict DIY-slot version.

This package keeps the Tencent platform contract:
- algorithm selection in UI: DIY
- runtime module path: agent_diy
"""


from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "upload_diy_minimal"


def reset_target() -> None:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    reset_target()
    copy_tree(ROOT / "agent_diy", TARGET / "agent_diy")
    copy_file(ROOT / "conf" / "algo_conf_intelligent_traffic_lights.toml", TARGET / "conf" / "algo_conf_intelligent_traffic_lights.toml")
    copy_file(ROOT / "conf" / "app_conf_intelligent_traffic_lights.toml", TARGET / "conf" / "app_conf_intelligent_traffic_lights.toml")
    copy_file(ROOT / "conf" / "configure_app.toml", TARGET / "conf" / "configure_app.toml")
    copy_file(ROOT / "kaiwu.json", TARGET / "kaiwu.json")
    copy_file(ROOT / "train_test.py", TARGET / "train_test.py")
    copy_file(ROOT / "DIY最小上传说明.md", TARGET / "DIY最小上传说明.md")
    print(f"Built DIY minimal upload package: {TARGET}")


if __name__ == "__main__":
    main()
