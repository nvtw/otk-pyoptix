# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Newton path-tracing viewer integration.

Newton is intentionally not imported by :mod:`warp_optix` itself. Import this
module explicitly after installing the ``newton`` extra.
"""

try:
    import newton as _newton  # noqa: F401
except ImportError as error:
    raise ImportError(
        "The warp_optix Newton viewer requires Newton 1.6.x. Install "
        "warp_optix[pathtracing,ui,newton] before importing it."
    ) from error

from .viewer import ViewerOptix

__all__ = ["ViewerOptix"]
