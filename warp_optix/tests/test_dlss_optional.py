# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from warp_optix import native as optix
import pytest


def test_dlss_support_query():
    # Using DLSS is optional, but every supported wheel includes the bindings.
    assert optix.dlss_support_available() is True


def test_dlss_contexts_share_sdk_lifetime():
    first = optix.DlssRRContext()
    second = optix.DlssRRContext()
    try:
        try:
            first.init()
        except RuntimeError as error:
            pytest.skip(f"DLSS runtime is unavailable: {error}")
        second.init()
        first.deinit()
        assert isinstance(second.isDlssRRAvailable(), bool)
    finally:
        second.deinit()
        first.deinit()
