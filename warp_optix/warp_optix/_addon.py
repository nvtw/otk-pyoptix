# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""warp_optix addon registration."""

from __future__ import annotations

from pathlib import Path


_REGISTERED = False


def _addon_include_dirs() -> list[str]:
    here = Path(__file__).resolve().parent
    include_dirs = [str(here / "_native" / "include")]
    try:
        import optix  # noqa: PLC0415

        include_dirs.append(optix.get_optix_include_dir())
    except Exception:
        # otk-pyoptix may be imported before its extension package is available.
        pass
    return include_dirs


def _addon_device_preamble() -> str:
    return '#include "warp_optix_builtins.h"\n'


def get_module_build_options(wp, existing=None):
    """Return module build options with warp_optix headers and preamble appended."""
    if existing is None:
        existing = wp.ModuleBuildOptions()
    if not isinstance(existing, wp.ModuleBuildOptions):
        raise TypeError(
            "extra_build_options must be a warp.ModuleBuildOptions instance "
            f"or None, got {type(existing).__name__}"
        )

    extra_include_dirs = list(existing.extra_include_dirs)
    for include_dir in _addon_include_dirs():
        if include_dir not in extra_include_dirs:
            extra_include_dirs.append(include_dir)

    extra_device_preamble = existing.extra_device_preamble
    addon_preamble = _addon_device_preamble()
    if addon_preamble not in extra_device_preamble:
        if extra_device_preamble and not extra_device_preamble.endswith("\n"):
            extra_device_preamble += "\n"
        extra_device_preamble += addon_preamble

    return wp.ModuleBuildOptions(
        extra_include_dirs=extra_include_dirs,
        extra_cuda_include_dirs=existing.extra_cuda_include_dirs,
        extra_cpu_include_dirs=existing.extra_cpu_include_dirs,
        extra_device_preamble=extra_device_preamble,
        extra_cpu_preamble=existing.extra_cpu_preamble,
    )


def register_with_warp() -> None:
    """Register warp_optix builtins with Warp."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    from warp_optix._builtins import register_addon_builtins  # noqa: PLC0415

    register_addon_builtins()
