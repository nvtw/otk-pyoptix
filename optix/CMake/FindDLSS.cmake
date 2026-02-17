if (TARGET DLSS::DLSS)
  return()
endif()

option(PYOPTIX_AUTO_DOWNLOAD_DLSS
  "Automatically download DLSS SDK when DLSS_ROOT is not set."
  ON
)
set(PYOPTIX_DLSS_VERSION "310.5.3" CACHE STRING "DLSS SDK version to fetch.")
set(PYOPTIX_DLSS_URL
  "https://github.com/NVIDIA/DLSS/archive/refs/tags/v${PYOPTIX_DLSS_VERSION}.tar.gz"
  CACHE STRING "DLSS SDK archive URL."
)
set(PYOPTIX_DLSS_URL_HASH
  "6b54a684b5b31e819a51742ad534abb4e8cdada76572f061a5d3149c7432a0a1"
  CACHE STRING "SHA256 for DLSS SDK archive."
)
set(PYOPTIX_DLSS_DIR
  "${CMAKE_SOURCE_DIR}/third_party/dlss-${PYOPTIX_DLSS_VERSION}"
  CACHE PATH "Directory used to store downloaded DLSS SDK."
)

set(_dlss_root "${DLSS_ROOT}")
if (NOT _dlss_root)
  set(_dlss_root "${PYOPTIX_DLSS_DIR}")
endif()

if (NOT EXISTS "${_dlss_root}/include/nvsdk_ngx.h" AND PYOPTIX_AUTO_DOWNLOAD_DLSS)
  set(_dlss_archive_dir "${PYOPTIX_DLSS_DIR}/_download")
  set(_dlss_archive_path "${_dlss_archive_dir}/dlss-${PYOPTIX_DLSS_VERSION}.tar.gz")
  set(_dlss_extract_dir "${_dlss_archive_dir}/src")

  message(STATUS "Downloading DLSS SDK v${PYOPTIX_DLSS_VERSION} from ${PYOPTIX_DLSS_URL}")
  file(MAKE_DIRECTORY "${_dlss_archive_dir}")
  file(DOWNLOAD "${PYOPTIX_DLSS_URL}" "${_dlss_archive_path}"
      STATUS _dlss_download_status
      SHOW_PROGRESS
      EXPECTED_HASH SHA256=${PYOPTIX_DLSS_URL_HASH}
  )
  list(GET _dlss_download_status 0 _dlss_download_status_code)
  if (NOT _dlss_download_status_code EQUAL 0)
    list(GET _dlss_download_status 1 _dlss_download_status_message)
    message(FATAL_ERROR
      "Failed to download DLSS SDK from ${PYOPTIX_DLSS_URL}: ${_dlss_download_status_message}. "
      "Set DLSS_ROOT manually to a local DLSS SDK path."
    )
  endif()

  file(REMOVE_RECURSE "${_dlss_extract_dir}")
  file(MAKE_DIRECTORY "${_dlss_extract_dir}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E tar xvf "${_dlss_archive_path}"
    WORKING_DIRECTORY "${_dlss_extract_dir}"
    RESULT_VARIABLE _dlss_extract_result
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if (NOT _dlss_extract_result EQUAL 0)
    message(FATAL_ERROR
      "Failed to extract DLSS SDK archive: ${_dlss_archive_path}. "
      "Set DLSS_ROOT manually to a local DLSS SDK path."
    )
  endif()

  file(GLOB _dlss_header_candidates "${_dlss_extract_dir}/*/include/nvsdk_ngx.h")
  list(LENGTH _dlss_header_candidates _dlss_header_candidates_len)
  if (_dlss_header_candidates_len EQUAL 0)
    message(FATAL_ERROR
      "Could not locate include/nvsdk_ngx.h in DLSS archive ${_dlss_archive_path}. "
      "Set DLSS_ROOT manually to a local DLSS SDK path."
    )
  endif()

  list(GET _dlss_header_candidates 0 _dlss_header_path)
  get_filename_component(_dlss_include_dir "${_dlss_header_path}" DIRECTORY)
  get_filename_component(_dlss_source_root "${_dlss_include_dir}" DIRECTORY)

  file(MAKE_DIRECTORY "${PYOPTIX_DLSS_DIR}")
  file(COPY "${_dlss_source_root}/include" DESTINATION "${PYOPTIX_DLSS_DIR}")
  if (EXISTS "${_dlss_source_root}/lib")
    file(COPY "${_dlss_source_root}/lib" DESTINATION "${PYOPTIX_DLSS_DIR}")
  endif()

  set(_dlss_root "${PYOPTIX_DLSS_DIR}")
  message(STATUS "Downloaded DLSS SDK to ${PYOPTIX_DLSS_DIR}")
endif()

set(DLSS_INCLUDE_DIR "${_dlss_root}/include")

if (WIN32)
  set(_dlss_crt_flavor "d")
  if (CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "MultiThreaded[^D]*$")
    set(_dlss_crt_flavor "s")
  endif()

  set(DLSS_LIBRARY_RELEASE "${_dlss_root}/lib/Windows_x86_64/x64/nvsdk_ngx_${_dlss_crt_flavor}.lib")
  set(DLSS_LIBRARY_DEBUG "${_dlss_root}/lib/Windows_x86_64/x64/nvsdk_ngx_${_dlss_crt_flavor}_dbg.lib")
  if (NOT EXISTS "${DLSS_LIBRARY_DEBUG}")
    set(DLSS_LIBRARY_DEBUG "${DLSS_LIBRARY_RELEASE}")
  endif()
  set(DLSS_RUNTIME_LIBRARY_DEBUG "${_dlss_root}/lib/Windows_x86_64/dev/nvngx_dlssd.dll")
  set(DLSS_RUNTIME_LIBRARY_RELEASE "${_dlss_root}/lib/Windows_x86_64/rel/nvngx_dlssd.dll")
else()
  set(DLSS_LIBRARY_RELEASE "${_dlss_root}/lib/Linux_x86_64/libnvsdk_ngx.a")
  set(DLSS_LIBRARY_DEBUG "${DLSS_LIBRARY_RELEASE}")
  file(GLOB _dlss_dev_so "${_dlss_root}/lib/Linux_x86_64/dev/libnvidia-ngx-dlssd.so*")
  file(GLOB _dlss_rel_so "${_dlss_root}/lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.so*")
  if (_dlss_dev_so)
    list(GET _dlss_dev_so 0 DLSS_RUNTIME_LIBRARY_DEBUG)
  endif()
  if (_dlss_rel_so)
    list(GET _dlss_rel_so 0 DLSS_RUNTIME_LIBRARY_RELEASE)
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DLSS
  FOUND_VAR DLSS_FOUND
  REQUIRED_VARS DLSS_INCLUDE_DIR DLSS_LIBRARY_RELEASE
  REASON_FAILURE_MESSAGE "DLSS SDK not found. Set DLSS_ROOT or enable PYOPTIX_AUTO_DOWNLOAD_DLSS."
)
if (NOT DLSS_FOUND)
  return()
endif()

if (NOT TARGET DLSS::DLSS)
  add_library(DLSS::DLSS UNKNOWN IMPORTED)
  set_target_properties(DLSS::DLSS PROPERTIES
    IMPORTED_CONFIGURATIONS "Debug;Release;RelWithDebInfo"
    INTERFACE_INCLUDE_DIRECTORIES "${DLSS_INCLUDE_DIR}"
    IMPORTED_LOCATION_RELEASE "${DLSS_LIBRARY_RELEASE}"
    IMPORTED_LOCATION_RELWITHDEBINFO "${DLSS_LIBRARY_RELEASE}"
    IMPORTED_LOCATION_DEBUG "${DLSS_LIBRARY_DEBUG}"
  )
endif()

function(dlss_setup_runtime_dependencies TARGET_NAME)
  if (NOT DLSS_RUNTIME_LIBRARY_DEBUG AND NOT DLSS_RUNTIME_LIBRARY_RELEASE)
    return()
  endif()
  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<IF:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,${DLSS_RUNTIME_LIBRARY_DEBUG},${DLSS_RUNTIME_LIBRARY_RELEASE}>"
    "$<TARGET_FILE_DIR:${TARGET_NAME}>/$<IF:$<PLATFORM_ID:Windows>,nvngx_dlssd.dll,libnvidia-ngx-dlssd.so>"
    COMMENT "Copying DLSS runtime library"
  )
endfunction()
