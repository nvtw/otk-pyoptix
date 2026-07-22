# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Python bindings for NVIDIA OptiX ray tracing SDK."""

import os
import sys
from pathlib import Path

__version__ = "0.1.0"
__author__ = "Keith Morley"
__license__ = "BSD-3-Clause"

_dll_directory_handles = []


def _add_dll_directories():
    """Make CUDA and packaged DLSS DLLs discoverable on Python 3.8+."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    package_dir = Path(__file__).resolve().parent
    _dll_directory_handles.append(os.add_dll_directory(str(package_dir)))

    cuda_bin = os.environ.get("CUDA_BIN_DIR")
    if cuda_bin and Path(cuda_bin).is_dir():
        _dll_directory_handles.append(os.add_dll_directory(cuda_bin))
        return

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and (Path(cuda_path) / "bin").is_dir():
        _dll_directory_handles.append(os.add_dll_directory(str(Path(cuda_path) / "bin")))


def get_optix_include_dir():
    """Return the packaged OptiX header directory."""
    include_dir = Path(__file__).resolve().parent / "include"
    if not (include_dir / "optix.h").exists():
        raise FileNotFoundError(f"Packaged OptiX headers were not found at '{include_dir}'.")
    return str(include_dir)


_add_dll_directories()

# Import everything from the native module.
from ._optix import *  # noqa: E402,F403

# Export all public symbols from _optix
__all__ = [name for name in dir() if not name.startswith('_')]
