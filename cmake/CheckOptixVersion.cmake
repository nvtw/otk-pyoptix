# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(warp_optix_check_optix_version header expected_version)
  if(NOT EXISTS "${header}")
    message(FATAL_ERROR "OptiX SDK header is missing: ${header}")
  endif()
  file(STRINGS "${header}" _version_line
       REGEX "^[ \t]*#[ \t]*define[ \t]+OPTIX_VERSION[ \t]+[0-9]+")
  string(REGEX MATCH "OPTIX_VERSION[ \t]+([0-9]+)" _match "${_version_line}")
  if(NOT _match)
    message(FATAL_ERROR "Cannot determine OptiX SDK version from ${header}")
  endif()
  set(_sdk_version "${CMAKE_MATCH_1}")
  math(EXPR _major "${_sdk_version} / 10000")
  math(EXPR _minor "(${_sdk_version} % 10000) / 100")
  math(EXPR _patch "${_sdk_version} % 100")
  set(_actual_version "${_major}.${_minor}.${_patch}")
  if(NOT "${_actual_version}" STREQUAL "${expected_version}")
    message(FATAL_ERROR
      "OptiX SDK version mismatch: headers are ${_actual_version}, but "
      "pyoptix==${expected_version} requires OptiX ${expected_version}. "
      "Use matching headers and clear any FETCHCONTENT_SOURCE_DIR_OPTIX_HEADERS override. "
      "Header: ${header}")
  endif()
endfunction()
