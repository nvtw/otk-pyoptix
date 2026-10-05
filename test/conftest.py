# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))

# Test-only compatibility for unchanged upstream test imports.
from warp_optix._backend import get_backend
sys.modules["optix"] = get_backend()
