# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python startup hook installed transiently by optix_newton_launcher.py."""

import importlib.util
import os


if (
    os.environ.get("WARP_OPTIX_NEWTON_LAUNCHER") == "1"
    and importlib.util.find_spec("newton") is not None
):
    from warp_optix.integrations.newton.launcher import activate

    activate()
