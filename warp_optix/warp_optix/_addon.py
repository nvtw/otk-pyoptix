# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""warp_optix addon registration."""

from __future__ import annotations

from pathlib import Path

from ._compat import _find_cuda_include_dir

_REGISTERED = False


def _addon_include_dirs() -> list[str]:
    here = Path(__file__).resolve().parent
    include_dirs = [str(here / "_native" / "include")]
    try:
        import optix

        optix_include_dir = optix.get_optix_include_dir()
    except (ImportError, AttributeError, OSError):
        # otk-pyoptix may be imported before its extension package is available.
        optix_include_dir = None
    if optix_include_dir is not None:
        include_dirs.append(optix_include_dir)
    if cuda_include_dir := _find_cuda_include_dir():
        include_dirs.append(str(cuda_include_dir))
    return include_dirs


def _addon_cuda_preamble() -> str:
    # Warp's generated source defines function-like ``float`` and ``int``
    # macros before injecting addon preambles. CUDA's half headers declare
    # conversion operators with those names, so suspend the macros while the
    # OptiX headers are parsed and restore them for Warp-generated code.
    return (
        "#undef float\n"
        "#undef int\n"
        '#include "warp_optix_builtins.h"\n'
        "#define float(x) cast_float(x)\n"
        "#define int(x) cast_int(x)\n"
    )


def _addon_build_dependencies() -> list[str]:
    here = Path(__file__).resolve().parent
    return [str(here / "_native" / "include" / "warp_optix_builtins.h")]


def get_module_build_options(wp, existing=None):
    """Return module build options with warp_optix headers and preamble appended."""
    if existing is None:
        existing = wp.ModuleBuildOptions()
    if not isinstance(existing, wp.ModuleBuildOptions):
        raise TypeError(
            "extra_build_options must be a warp.ModuleBuildOptions instance "
            f"or None, got {type(existing).__name__}"
        )

    extra_cuda_include_dirs = list(existing.extra_cuda_include_dirs)
    for include_dir in _addon_include_dirs():
        if include_dir not in extra_cuda_include_dirs:
            extra_cuda_include_dirs.append(include_dir)

    extra_cuda_preamble = existing.extra_cuda_preamble
    addon_preamble = _addon_cuda_preamble()
    if addon_preamble not in extra_cuda_preamble:
        if extra_cuda_preamble and not extra_cuda_preamble.endswith("\n"):
            extra_cuda_preamble += "\n"
        extra_cuda_preamble += addon_preamble

    extra_build_dependencies = list(existing.extra_build_dependencies)
    for dependency in _addon_build_dependencies():
        if dependency not in extra_build_dependencies:
            extra_build_dependencies.append(dependency)

    return wp.ModuleBuildOptions(
        extra_cuda_include_dirs=extra_cuda_include_dirs,
        extra_cpu_include_dirs=existing.extra_cpu_include_dirs,
        extra_cuda_preamble=extra_cuda_preamble,
        extra_cpu_preamble=existing.extra_cpu_preamble,
        extra_build_dependencies=extra_build_dependencies,
    )


def register_with_warp() -> None:
    """Register warp_optix builtins with Warp."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    from warp_optix._builtins import register_addon_builtins

    register_addon_builtins()
