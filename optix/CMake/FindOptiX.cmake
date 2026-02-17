#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

if (TARGET OptiX::OptiX)
  return()
endif()

option(PYOPTIX_AUTO_DOWNLOAD_OPTIX_HEADERS
  "Automatically download OptiX headers from NVIDIA/optix-dev when OptiX_INSTALL_DIR is not set."
  ON
)
set(PYOPTIX_OPTIX_HEADERS_VERSION "9.0.0" CACHE STRING "OptiX headers version to fetch from NVIDIA/optix-dev.")
set(PYOPTIX_OPTIX_HEADERS_DIR
  "${CMAKE_SOURCE_DIR}/third_party/optix-dev-${PYOPTIX_OPTIX_HEADERS_VERSION}"
  CACHE PATH "Directory used to store downloaded OptiX headers."
)

macro(OptiX_config_message)
  if (NOT DEFINED OptiX_FIND_QUIETLY)
    message(${ARGN})
  endif()
endmacro()

# Locate the OptiX distribution.  Search relative to the SDK first, then look in the system.

if (NOT OptiX_INSTALL_DIR AND PYOPTIX_AUTO_DOWNLOAD_OPTIX_HEADERS)
  if (EXISTS "${PYOPTIX_OPTIX_HEADERS_DIR}/include/optix.h")
    OptiX_config_message(STATUS "Using cached OptiX headers from ${PYOPTIX_OPTIX_HEADERS_DIR}")
    set(OptiX_INSTALL_DIR "${PYOPTIX_OPTIX_HEADERS_DIR}")
  else()
    set(_optix_headers_url "https://github.com/NVIDIA/optix-dev/archive/refs/tags/v${PYOPTIX_OPTIX_HEADERS_VERSION}.zip")
    set(_optix_archive_dir "${PYOPTIX_OPTIX_HEADERS_DIR}/_download")
    set(_optix_archive_path "${_optix_archive_dir}/optix-dev-v${PYOPTIX_OPTIX_HEADERS_VERSION}.zip")
    set(_optix_extract_dir "${_optix_archive_dir}/src")

    OptiX_config_message(STATUS "Downloading OptiX headers v${PYOPTIX_OPTIX_HEADERS_VERSION} from ${_optix_headers_url}")
    file(MAKE_DIRECTORY "${_optix_archive_dir}")
    file(DOWNLOAD "${_optix_headers_url}" "${_optix_archive_path}" STATUS _optix_download_status SHOW_PROGRESS)
    list(GET _optix_download_status 0 _optix_download_status_code)
    if (NOT _optix_download_status_code EQUAL 0)
      list(GET _optix_download_status 1 _optix_download_status_message)
      message(FATAL_ERROR
        "Failed to download OptiX headers from ${_optix_headers_url}: ${_optix_download_status_message}. "
        "Set OptiX_INSTALL_DIR manually to a local OptiX SDK/install path."
      )
    endif()

    file(REMOVE_RECURSE "${_optix_extract_dir}")
    file(MAKE_DIRECTORY "${_optix_extract_dir}")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E tar xvf "${_optix_archive_path}"
      WORKING_DIRECTORY "${_optix_extract_dir}"
      RESULT_VARIABLE _optix_extract_result
      OUTPUT_QUIET
      ERROR_QUIET
    )
    if (NOT _optix_extract_result EQUAL 0)
      message(FATAL_ERROR
        "Failed to extract downloaded OptiX headers archive: ${_optix_archive_path}. "
        "Set OptiX_INSTALL_DIR manually to a local OptiX SDK/install path."
      )
    endif()

    file(GLOB _optix_header_candidates "${_optix_extract_dir}/*/include/optix.h")
    list(LENGTH _optix_header_candidates _optix_header_candidates_len)
    if (_optix_header_candidates_len EQUAL 0)
      message(FATAL_ERROR
        "Could not locate include/optix.h in downloaded archive ${_optix_archive_path}. "
        "Set OptiX_INSTALL_DIR manually to a local OptiX SDK/install path."
      )
    endif()

    list(GET _optix_header_candidates 0 _optix_header_path)
    get_filename_component(_optix_include_dir "${_optix_header_path}" DIRECTORY)
    get_filename_component(_optix_source_root "${_optix_include_dir}" DIRECTORY)

    file(MAKE_DIRECTORY "${PYOPTIX_OPTIX_HEADERS_DIR}")
    file(COPY "${_optix_source_root}/include" DESTINATION "${PYOPTIX_OPTIX_HEADERS_DIR}")

    if (EXISTS "${_optix_source_root}/LICENSE.txt")
      file(COPY "${_optix_source_root}/LICENSE.txt" DESTINATION "${PYOPTIX_OPTIX_HEADERS_DIR}")
    endif()
    if (EXISTS "${_optix_source_root}/README.md")
      file(COPY "${_optix_source_root}/README.md" DESTINATION "${PYOPTIX_OPTIX_HEADERS_DIR}")
    endif()

    set(OptiX_INSTALL_DIR "${PYOPTIX_OPTIX_HEADERS_DIR}")
    OptiX_config_message(STATUS "Downloaded OptiX headers to ${PYOPTIX_OPTIX_HEADERS_DIR}")
  endif()
endif()

find_path(OptiX_ROOT_DIR NAMES include/optix.h PATHS "${OptiX_INSTALL_DIR}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
  FOUND_VAR OptiX_FOUND
  REQUIRED_VARS
    OptiX_ROOT_DIR
  REASON_FAILURE_MESSAGE
    "OptiX installation not found on CMAKE_PREFIX_PATH (include/optix.h)"
)

if (NOT OptiX_FOUND)
  set(OptiX_NOT_FOUND_MESSAGE "Unable to find OptiX, please add your OptiX installation to CMAKE_PREFIX_PATH")
  return()
endif()

set(OptiX_INCLUDE_DIR ${OptiX_ROOT_DIR}/include)

add_library(OptiX::OptiX INTERFACE IMPORTED)
target_include_directories(OptiX::OptiX INTERFACE ${OptiX_INCLUDE_DIR})

