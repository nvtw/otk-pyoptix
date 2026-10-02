# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Python bindings for NVIDIA OptiX ray tracing SDK."""

import os
import sys
import sysconfig
from pathlib import Path

__version__ = "9.1.0"
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
from . import _optix as _native
from ._optix import *  # noqa: E402,F403


def dlss_support_available():
    """Return whether this wheel was compiled with DLSS Ray Reconstruction."""
    return hasattr(_native, "DlssRRContext")


def dlss_rr_available():
    """Return whether DLSS Ray Reconstruction can run on this system."""
    if not dlss_support_available():
        return False
    context = DlssRRContext()
    try:
        context.init(featureSearchPath=str(Path(__file__).resolve().parent))
        return context.isDlssRRAvailable()
    except RuntimeError:
        return False
    finally:
        context.deinit()


if not dlss_support_available():
    from . import _dlss_stub as _stub

    for _name in _stub.__all__:
        globals()[_name] = getattr(_stub, _name)
    for _enum in (_stub.DlssPerfQuality, _stub.RayReconstructionHintRenderPreset, _stub.DlssRRResource):
        for _member in _enum:
            globals()[_member.name] = _member

# Export all public symbols from _optix
__all__ = [name for name in dir() if not name.startswith('_')]
