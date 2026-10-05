# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Access the bindings shipped with this distribution, never standalone PyOptiX."""

from importlib import import_module


def get_backend():
    try:
        return import_module("warp_optix._bindings")
    except ImportError as error:
        raise ImportError(
            "The bundled warp-optix bindings could not be loaded. Install a compatible "
            "warp-optix wheel or build this repository with `pip install .`."
        ) from error
