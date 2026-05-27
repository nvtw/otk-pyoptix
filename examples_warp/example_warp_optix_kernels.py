# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal Warp-defined OptiX entry kernels.

This example defines OptiX raygen/miss entry points with the new `kernel_type` API,
then emits CUDA source so the generated entry names can be inspected.
"""

import warp as wp
import warp_optix as woptix


@woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
def raygen_program():
    launch_idx = wp.optix_get_launch_index()
    if launch_idx[0] == 0 and launch_idx[1] == 0 and launch_idx[2] == 0:
        pass


@woptix.optix_kernel(woptix.OptixKernelType.MISS)
def miss_program():
    return


def main() -> None:
    module = wp.get_module(__name__)
    hasher = wp._src.context.ModuleHasher(module._get_live_kernels(), module.options.copy())
    builder = wp._src.context.ModuleBuilder(module, options=module.options.copy(), hasher=hasher)
    cuda_source = builder.codegen("cuda")

    print("Generated OptiX entry symbols:")
    print(f"  __raygen__{raygen_program.get_mangled_name()}")
    print(f"  __miss__{miss_program.get_mangled_name()}")
    print("\nFirst 40 lines of generated CUDA source:")
    print("\n".join(cuda_source.splitlines()[:40]))


if __name__ == "__main__":
    main()
