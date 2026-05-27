"""warp_optix addon registration.

Importing :mod:`warp_optix` calls :func:`register_with_warp`, which appends
this package's include directories and device preamble to warp's existing
``warp.config`` knobs and registers the OptiX-named builtins via warp's
existing :func:`warp.add_builtin` public API. There is no warp-side registration
framework — just three small config attributes that warp consults during CUDA
compile.
"""

from __future__ import annotations

from pathlib import Path


_REGISTERED = False


def _addon_include_dirs() -> list[str]:
    here = Path(__file__).resolve().parent
    include_dirs: list[str] = [str(here / "_native" / "include")]
    try:
        import optix  # noqa: PLC0415
        include_dirs.append(optix.get_optix_include_dir())
    except Exception:
        # otk-pyoptix not importable or not built with headers; the user can
        # still set OPTIX_INCLUDE_DIR or extra_include_dirs themselves.
        pass
    return include_dirs


def register_with_warp() -> None:
    """Wire warp_optix into warp.

    Idempotent. Called automatically on ``import warp_optix``.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    import warp as wp  # noqa: PLC0415

    # 1) Include directories (OptiX SDK headers + warp_optix_builtins.h).
    extra_dirs = getattr(wp.config, "extra_include_dirs", None)
    if isinstance(extra_dirs, list):
        for d in _addon_include_dirs():
            if d not in extra_dirs:
                extra_dirs.append(d)

    # 2) Device preamble: pull in warp_optix_builtins.h from every CUDA TU.
    preamble = getattr(wp.config, "extra_device_preamble", None)
    extra = '#include "warp_optix_builtins.h"\n'
    if isinstance(preamble, str) and extra not in preamble:
        wp.config.extra_device_preamble = preamble + extra

    # 3) OptiX-named builtins (uses warp's existing public add_builtin).
    from warp_optix._builtins import register_addon_builtins  # noqa: PLC0415
    register_addon_builtins()
