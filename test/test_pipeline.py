# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.



import optix
import pytest 
import cupy as cp

import tutil 




class TestPipeline:

    def test_pipeline_options( self ):

        pipeline_options = optix.PipelineCompileOptions()
        pipeline_options.usesMotionBlur        = False
        pipeline_options.traversableGraphFlags = optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
        pipeline_options.numPayloadValues      = 2
        pipeline_options.numAttributeValues    = 2
        pipeline_options.exceptionFlags        = optix.EXCEPTION_FLAG_NONE
        pipeline_options.pipelineLaunchParamsVariableName = "params1"
        assert pipeline_options.pipelineLaunchParamsVariableName == "params1"


        pipeline_options = optix.PipelineCompileOptions(
            usesMotionBlur        = False,
            traversableGraphFlags = optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
            numPayloadValues      = 3,
            numAttributeValues    = 4,
            exceptionFlags        = optix.EXCEPTION_FLAG_NONE,
            pipelineLaunchParamsVariableName = "params2"
            )
        assert pipeline_options.pipelineLaunchParamsVariableName == "params2"

    def test_builtin_curve_primitive_enums(self):
        assert int(optix.PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER) == 0x2507
        assert int(optix.PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BEZIER) == 1 << 7
        assert int(optix.PRIMITIVE_TYPE_ROUND_CATMULLROM) == 0x2504
        assert int(optix.PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM) == 1 << 4
        assert int(optix.PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE) == 0x2505
        assert int(optix.PRIMITIVE_TYPE_FLAGS_FLAT_QUADRATIC_BSPLINE) == 1 << 5
