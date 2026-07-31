# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Python bindings for NVIDIA OptiX ray tracing SDK."""

import os
import sys
import sysconfig
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
    package_include = Path(__file__).resolve().parent / "include"
    include_dirs = [package_include]

    # Editable installs execute this source-tree __init__.py while CMake places
    # generated/fetched package data in the environment's platform library.
    # Check that location as well so get_optix_include_dir() behaves identically
    # for editable and wheel installs.
    platlib_include = Path(sysconfig.get_path("platlib")) / "optix" / "include"
    if platlib_include != package_include:
        include_dirs.append(platlib_include)

    for include_dir in include_dirs:
        if (include_dir / "optix.h").is_file():
            return str(include_dir)

    searched = "\n  - ".join(str(path) for path in include_dirs)
    raise FileNotFoundError(f"Packaged OptiX headers were not found in:\n  - {searched}")


_add_dll_directories()

# Import everything from the native module.
from ._optix import *  # noqa: E402,F403

# Export all public symbols from _optix
__all__ = [name for name in dir() if not name.startswith('_')]
