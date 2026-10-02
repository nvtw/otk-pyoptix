#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an unchanged Newton command with warp_optix's OptiX viewer.

Example:
    ./optix_newton_launcher.py uv run --extra examples python -m newton.examples basic_pendulum
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepend_pythonpath(environment: dict[str, str], *paths: Path) -> None:
    entries = [str(path) for path in paths]
    existing = environment.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} COMMAND [ARG ...]")

    repository_root = Path(__file__).resolve().parent
    package_root = repository_root / "warp_optix"
    startup_hook = package_root / "warp_optix" / "integrations" / "newton" / "_launcher_site"
    environment = os.environ.copy()
    environment["WARP_OPTIX_NEWTON_LAUNCHER"] = "1"
    _prepend_pythonpath(environment, startup_hook, package_root)

    command = sys.argv[1:]
    try:
        os.execvpe(command[0], command, environment)
    except FileNotFoundError as error:
        raise SystemExit(f"{Path(sys.argv[0]).name}: command not found: {command[0]}") from error


if __name__ == "__main__":
    main()
