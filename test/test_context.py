# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import cupy as cp
import optix as ox
import pytest 

import tutil 


class Logger:
    def __init__(self):
        self.num_mssgs = 0
    
    def __call__(self, level, tag, mssg):
        print("[{:>2}][{:>12}]: {}".format(level, tag, mssg))
        self.num_mssgs += 1
    

def log_callback(level, tag, mssg):
    print("[{:>2}][{:>12}]: {}".format(level, tag, mssg))


class TestDeviceContextOptions:
    def test_default_ctor(self):
        options = ox.DeviceContextOptions()
        assert options.logCallbackFunction is None
        assert options.logCallbackLevel == 0
        if tutil.optix_version_gte( (7,2) ): 
            assert options.validationMode == ox.DEVICE_CONTEXT_VALIDATION_MODE_OFF

    def test_ctor0(self):
        options = ox.DeviceContextOptions(log_callback)
        assert options.logCallbackFunction == log_callback
   
    def test_ctor1(self):
        logger = Logger()
        if tutil.optix_version_gte( (7,2) ): 
            options = ox.DeviceContextOptions(
                logCallbackFunction = logger,
                logCallbackLevel    = 3,
                validationMode      = ox.DEVICE_CONTEXT_VALIDATION_MODE_ALL
            )
        else:
            options = ox.DeviceContextOptions(
                logCallbackFunction = logger,
                logCallbackLevel    = 3
            )
        assert options.logCallbackFunction == logger
        assert options.logCallbackLevel    == 3
        if tutil.optix_version_gte( (7,2) ): 
            assert options.validationMode == ox.DEVICE_CONTEXT_VALIDATION_MODE_ALL
        else:
            assert options.validationMode == ox.DEVICE_CONTEXT_VALIDATION_MODE_OFF

    def test_context_options_props(self):
        options = ox.DeviceContextOptions()
        options.logCallbackLevel = 1
        assert options.logCallbackLevel == 1

        options.logCallbackFunction = log_callback
        assert options.logCallbackFunction == log_callback 


@pytest.mark.skipif(not tutil.optix_version_gte((9, 0)), reason="requires OptiX 9")
class TestCoopVecTypes:
    def test_matrix_description_defaults_and_properties(self):
        desc = ox.CoopVecMatrixDescription()
        desc.N = 64
        desc.K = 48
        desc.offsetInBytes = 128
        desc.elementType = ox.COOP_VEC_ELEM_TYPE_FLOAT8_E4M3
        desc.layout = ox.COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR
        desc.rowColumnStrideInBytes = 0
        desc.sizeInBytes = 3072

        assert desc.N == 64
        assert desc.K == 48
        assert desc.offsetInBytes == 128
        assert desc.elementType == ox.COOP_VEC_ELEM_TYPE_FLOAT8_E4M3
        assert desc.layout == ox.COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR
        assert desc.rowColumnStrideInBytes == 0
        assert desc.sizeInBytes == 3072

    def test_coop_vec_enums_are_exported(self):
        assert int(ox.COOP_VEC_ELEM_TYPE_FLOAT16) != int(ox.COOP_VEC_ELEM_TYPE_FLOAT32)
        assert int(ox.COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR) != int(
            ox.COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL
        )


class TestContext:
    def test_create_destroy( self ):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        ctx.destroy()

    def test_get_property( self ):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        v = ctx.getProperty( ox.DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK )
        assert type( v ) is int
        assert v > 1 and v <= 16  # at time of writing, was 8
        ctx.destroy()

    @pytest.mark.skipif(not tutil.optix_version_gte((9, 0)), reason="requires OptiX 9")
    def test_coop_vec_property_and_matrix_size(self):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        try:
            flags = ctx.getProperty(ox.DEVICE_PROPERTY_COOP_VEC)
            assert isinstance(flags, int)
            if flags & int(ox.DEVICE_PROPERTY_COOP_VEC_FLAG_STANDARD):
                size = ctx.coopVecMatrixComputeSize(
                    64,
                    48,
                    ox.COOP_VEC_ELEM_TYPE_FLOAT8_E4M3,
                    ox.COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
                )
                assert size > 0
                assert size % 64 == 0
        finally:
            ctx.destroy()

    @pytest.mark.skipif(not tutil.optix_version_gte((9, 0)), reason="requires OptiX 9")
    def test_coop_vec_matrix_conversion_round_trip(self):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        try:
            flags = ctx.getProperty(ox.DEVICE_PROPERTY_COOP_VEC)
            if not flags & int(ox.DEVICE_PROPERTY_COOP_VEC_FLAG_STANDARD):
                pytest.skip("device does not support OptiX cooperative vectors")

            n = 16
            k = 16
            row_size = ctx.coopVecMatrixComputeSize(
                n, k, ox.COOP_VEC_ELEM_TYPE_FLOAT16, ox.COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR
            )
            optimal_size = ctx.coopVecMatrixComputeSize(
                n,
                k,
                ox.COOP_VEC_ELEM_TYPE_FLOAT16,
                ox.COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
            )

            def description(layout, size):
                desc = ox.CoopVecMatrixDescription()
                desc.N = n
                desc.K = k
                desc.offsetInBytes = 0
                desc.elementType = ox.COOP_VEC_ELEM_TYPE_FLOAT16
                desc.layout = layout
                desc.rowColumnStrideInBytes = 0
                desc.sizeInBytes = size
                return desc

            row_desc = description(ox.COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR, row_size)
            optimal_desc = description(
                ox.COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL, optimal_size
            )
            source = cp.arange(n * k, dtype=cp.float16)
            packed = cp.empty(optimal_size, dtype=cp.uint8)
            restored = cp.empty(row_size, dtype=cp.uint8)
            stream = cp.cuda.get_current_stream()

            ctx.coopVecMatrixConvert(
                stream.ptr,
                [row_desc],
                source.data.ptr,
                [optimal_desc],
                packed.data.ptr,
            )
            ctx.coopVecMatrixConvert(
                stream.ptr,
                [optimal_desc],
                packed.data.ptr,
                [row_desc],
                restored.data.ptr,
            )
            stream.synchronize()
            cp.testing.assert_array_equal(
                restored[: source.nbytes].view(cp.float16), source
            )
        finally:
            ctx.destroy()

    @pytest.mark.skipif(not tutil.optix_version_gte((9, 0)), reason="requires OptiX 9")
    def test_coop_vec_conversion_validates_before_calling_optix(self):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        try:
            with pytest.raises(ValueError, match="numNetworks"):
                ctx.coopVecMatrixConvert(0, [], 0, [], 0, numNetworks=0)
            with pytest.raises(ValueError, match="same length"):
                ctx.coopVecMatrixConvert(
                    0, [ox.CoopVecMatrixDescription()], 64, [], 64
                )

            desc = ox.CoopVecMatrixDescription()
            desc.N = desc.K = 16
            desc.elementType = ox.COOP_VEC_ELEM_TYPE_FLOAT16
            desc.layout = ox.COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR
            desc.sizeInBytes = 512
            with pytest.raises(ValueError, match="64-byte aligned"):
                ctx.coopVecMatrixConvert(0, [desc], 65, [desc], 128)
            with pytest.raises(ValueError, match="strides must be non-zero"):
                ctx.coopVecMatrixConvert(0, [desc], 64, [desc], 128, numNetworks=2)
        finally:
            ctx.destroy()
    
    def test_set_log_callback( self ):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        logger = Logger()
        ctx.setLogCallback( logger, 3 )
        ctx.setLogCallback( None, 2 )
        ctx.setLogCallback( log_callback, 1 )
        ctx.destroy()

    def test_cache_default(self):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        assert ctx.getCacheEnabled()
        ctx.destroy()

    def test_cache_enable_disable(self):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        ctx.setCacheEnabled(False);
        assert not ctx.getCacheEnabled()
        ctx.setCacheEnabled(True);
        assert ctx.getCacheEnabled()
        ctx.destroy()

    def test_cache_database_sizes(self):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        db_sizes = ( 1024, 1024*1024 )
        ctx.setCacheDatabaseSizes( *db_sizes )
        assert ctx.getCacheDatabaseSizes() == db_sizes 
        ctx.destroy()
        
    def test_set_get_cache( self ):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())

        v = ctx.getCacheLocation() 
        assert type(v) is str

        loc =  "/dev/null"
        with pytest.raises( RuntimeError ):
            ctx.setCacheLocation( loc ) # not valid dir
        ctx.destroy()



