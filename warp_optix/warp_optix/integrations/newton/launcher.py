# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup support for launching unmodified Newton examples with OptiX."""

from __future__ import annotations

import sys
from collections.abc import MutableSequence

from .viewer import ViewerOptix


def _force_gl_selection(argv: MutableSequence[str]) -> None:
    """Route an explicit Newton viewer selection through its gl branch."""
    for index, argument in enumerate(argv):
        if argument == "--viewer" and index + 1 < len(argv):
            argv[index + 1] = "gl"
        elif argument.startswith("--viewer="):
            argv[index] = "--viewer=gl"


def activate(argv: MutableSequence[str] | None = None) -> None:
    """Make Newton's public ViewerGL name construct ViewerOptix.

    Newton's example launcher defaults to the gl branch and constructs the
    public newton.viewer.ViewerGL class. Rebinding that name keeps the
    complete Newton command and example code unchanged while selecting the
    external OptiX implementation.
    """
    import newton.viewer

    if argv is None:
        argv = sys.argv
    _force_gl_selection(argv)
    newton.viewer.ViewerGL = ViewerOptix
