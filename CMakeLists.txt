# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 3.17...3.31)

# Use scikit-build-core provided project name and version
project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

#------------------------------------------------------------------------------
# Python and pybind11
#------------------------------------------------------------------------------

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#------------------------------------------------------------------------------
# Fetch OptiX headers from NVIDIA/optix-dev repository
#------------------------------------------------------------------------------

include(FetchContent)
include(CheckOptixVersion)

# Use the dependency pin as the SDK version for fetching and validation.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _project_metadata)
string(REGEX MATCH "\"pyoptix==([0-9]+\\.[0-9]+\\.[0-9]+)\""
       _pyoptix_requirement "${_project_metadata}")
if(NOT _pyoptix_requirement)
  message(FATAL_ERROR "warp-optix requires an exact pyoptix SDK version pin in pyproject.toml")
endif()
set(WARP_OPTIX_OPTIX_VERSION "${CMAKE_MATCH_1}")

message(STATUS "Fetching OptiX headers from NVIDIA/optix-dev...")
FetchContent_Declare(
    optix_headers
    GIT_REPOSITORY https://github.com/NVIDIA/optix-dev
    GIT_TAG        v${WARP_OPTIX_OPTIX_VERSION}
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(optix_headers)
warp_optix_check_optix_version("${optix_headers_SOURCE_DIR}/include/optix.h"
                              "${WARP_OPTIX_OPTIX_VERSION}")

# Create OptiX::OptiX interface target using fetched headers
add_library(OptiX::OptiX INTERFACE IMPORTED)
target_include_directories(OptiX::OptiX INTERFACE ${optix_headers_SOURCE_DIR}/include)

message(STATUS "Using OptiX headers from: ${optix_headers_SOURCE_DIR}/include")

# Find CUDA Toolkit (required for CUDA runtime)
if(NOT TARGET CUDA::cuda_driver)
  find_package(CUDAToolkit 10.0 REQUIRED)
endif()

# DLSS Ray Reconstruction is enabled by default. The environment-variable
# fallbacks keep the previous setup.py interface working after the migration to
# scikit-build-core; CMake cache arguments still take precedence.
if(NOT DEFINED PYOPTIX_ENABLE_DLSS AND DEFINED ENV{PYOPTIX_ENABLE_DLSS})
  set(PYOPTIX_ENABLE_DLSS "$ENV{PYOPTIX_ENABLE_DLSS}")
endif()
option(PYOPTIX_ENABLE_DLSS "Enable DLSS Ray Reconstruction bindings" ON)

foreach(_dlss_setting DLSS_ROOT PYOPTIX_DLSS_VERSION PYOPTIX_AUTO_DOWNLOAD_DLSS)
  if(NOT DEFINED ${_dlss_setting} AND DEFINED ENV{${_dlss_setting}})
    set(${_dlss_setting} "$ENV{${_dlss_setting}}" CACHE STRING "Forwarded from the environment")
  endif()
endforeach()

if(PYOPTIX_ENABLE_DLSS)
  find_package(DLSS REQUIRED)
endif()

#------------------------------------------------------------------------------
# Main build
#------------------------------------------------------------------------------

python_add_library(_ext MODULE src/extension.cpp WITH_SOABI)
set_target_properties(_ext PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_link_libraries(_ext PRIVATE
    OptiX::OptiX
    CUDA::cuda_driver
    CUDA::cudart
    pybind11::headers
)

if(PYOPTIX_ENABLE_DLSS)
  target_compile_definitions(_ext PRIVATE PYOPTIX_ENABLE_DLSS=1)
  target_link_libraries(_ext PRIVATE DLSS::DLSS)
  dlss_setup_runtime_dependencies(_ext)
endif()

target_compile_features(_ext PRIVATE cxx_std_17)

# The dependency owns optix._optix. Install only our supplemental extension,
# shader headers, and optional DLSS runtime under warp_optix.
set(WARP_OPTIX_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}/warp_optix")
install(TARGETS _ext DESTINATION ${WARP_OPTIX_INSTALL_DIR})
install(DIRECTORY "${optix_headers_SOURCE_DIR}/include/"
        DESTINATION ${WARP_OPTIX_INSTALL_DIR}/include)

if(PYOPTIX_ENABLE_DLSS AND (DLSS_RUNTIME_LIBRARY_DEBUG OR DLSS_RUNTIME_LIBRARY_RELEASE))
  if(NOT EXISTS "${DLSS_LICENSE_FILE}")
    message(FATAL_ERROR "DLSS runtime packaging requires the SDK LICENSE.txt in DLSS_ROOT")
  endif()
  install(FILES "${DLSS_LICENSE_FILE}" DESTINATION ${WARP_OPTIX_INSTALL_DIR}
          RENAME DLSS_LICENSE.txt)
  install(FILES
    "$<IF:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,${DLSS_RUNTIME_LIBRARY_DEBUG},${DLSS_RUNTIME_LIBRARY_RELEASE}>"
    DESTINATION ${WARP_OPTIX_INSTALL_DIR}
    RENAME "$<IF:$<PLATFORM_ID:Windows>,nvngx_dlssd.dll,libnvidia-ngx-dlssd.so>"
  )
endif()
