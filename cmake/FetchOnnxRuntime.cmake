# FetchOnnxRuntime.cmake
# Downloads official Microsoft ONNX Runtime binaries as a workaround for
# vcpkg build issues (see https://github.com/microsoft/vcpkg/issues/49349)
#
# Usage:
#   include(cmake/FetchOnnxRuntime.cmake)
#   fetch_onnxruntime(VERSION 1.23.2 DESTINATION ${CMAKE_BINARY_DIR}/onnxruntime)
#
# After calling, ONNXRUNTIME_DLL_DIR will be set to the directory containing DLLs

function(fetch_onnxruntime)
    # Added optional SHA256 verification for supply-chain safety.
    cmake_parse_arguments(ARG "" "VERSION;DESTINATION;SHA256" "" ${ARGN})

    if(NOT ARG_VERSION)
        set(ARG_VERSION "1.23.2")
    endif()

    if(NOT ARG_DESTINATION)
        set(ARG_DESTINATION "${CMAKE_BINARY_DIR}/onnxruntime-official")
    endif()

    set(ONNX_ARCHIVE_NAME "onnxruntime-win-x64-${ARG_VERSION}")
    set(ONNX_ARCHIVE_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ARG_VERSION}/${ONNX_ARCHIVE_NAME}.zip")
    set(ONNX_ARCHIVE_PATH "${ARG_DESTINATION}/${ONNX_ARCHIVE_NAME}.zip")
    set(ONNX_EXTRACT_DIR "${ARG_DESTINATION}/${ONNX_ARCHIVE_NAME}")
    set(ONNX_DLL_DIR "${ONNX_EXTRACT_DIR}/lib")

    # If a SHA256 was provided, verify existing archive before trusting it
    if(ARG_SHA256)
        if(EXISTS "${ONNX_ARCHIVE_PATH}")
            message(STATUS "Verifying checksum for existing ${ONNX_ARCHIVE_PATH}...")
            if(UNIX)
                execute_process(COMMAND sha256sum "${ONNX_ARCHIVE_PATH}"
                    OUTPUT_VARIABLE _SUM_OUT
                    RESULT_VARIABLE _SUM_RC
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
                string(REGEX MATCH "^[0-9a-fA-F]+" _COPY_HASH "${_SUM_OUT}")
            elseif(WIN32)
                execute_process(COMMAND certutil -hashfile "${ONNX_ARCHIVE_PATH}" SHA256
                    OUTPUT_VARIABLE _SUM_OUT
                    RESULT_VARIABLE _SUM_RC
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
                string(REGEX MATCH "[0-9A-Fa-f]{64}" _COPY_HASH "${_SUM_OUT}")
            else()
                set(_COPY_HASH "")
            endif()

            if(NOT _COPY_HASH STREQUAL "${ARG_SHA256}")
                message(WARNING "Checksum mismatch for ${ONNX_ARCHIVE_PATH} (expected ${ARG_SHA256}, got ${_COPY_HASH}). Re-downloading.")
                file(REMOVE "${ONNX_ARCHIVE_PATH}")
            else()
                message(STATUS "Checksum verified for ${ONNX_ARCHIVE_NAME}")
            endif()
        endif()
    endif()

    # Check if already downloaded and extracted
    if(EXISTS "${ONNX_DLL_DIR}/onnxruntime.dll")
        message(STATUS "ONNX Runtime ${ARG_VERSION} already available at ${ONNX_DLL_DIR}")
    else()
        message(STATUS "Downloading official ONNX Runtime ${ARG_VERSION}...")

        # Create destination directory
        file(MAKE_DIRECTORY ${ARG_DESTINATION})

        # Download if not present (optionally with EXPECTED_HASH)
        if(NOT EXISTS "${ONNX_ARCHIVE_PATH}")
            if(ARG_SHA256)
                file(DOWNLOAD
                    "${ONNX_ARCHIVE_URL}"
                    "${ONNX_ARCHIVE_PATH}"
                    SHOW_PROGRESS
                    EXPECTED_HASH "SHA256=${ARG_SHA256}"
                    STATUS DOWNLOAD_STATUS
                )
            else()
                file(DOWNLOAD
                    "${ONNX_ARCHIVE_URL}"
                    "${ONNX_ARCHIVE_PATH}"
                    SHOW_PROGRESS
                    STATUS DOWNLOAD_STATUS
                )
            endif()

            list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
            if(NOT STATUS_CODE EQUAL 0)
                message(FATAL_ERROR "Failed to download ONNX Runtime: ${DOWNLOAD_STATUS}")
            endif()
        endif()

        # Extract
        message(STATUS "Extracting ONNX Runtime...")
        file(ARCHIVE_EXTRACT
            INPUT "${ONNX_ARCHIVE_PATH}"
            DESTINATION "${ARG_DESTINATION}"
        )

        if(NOT EXISTS "${ONNX_DLL_DIR}/onnxruntime.dll")
            message(FATAL_ERROR "Failed to extract ONNX Runtime - DLL not found")
        endif()

        message(STATUS "ONNX Runtime ${ARG_VERSION} ready at ${ONNX_DLL_DIR}")
    endif()

    # Export the DLL directory and archive path to parent scope (useful for caching)
    set(ONNXRUNTIME_OFFICIAL_DLL_DIR "${ONNX_DLL_DIR}" PARENT_SCOPE)
    set(ONNXRUNTIME_OFFICIAL_ARCHIVE "${ONNX_ARCHIVE_PATH}" PARENT_SCOPE)
endfunction()

# Helper function to copy official ONNX Runtime DLLs to a target's output directory
function(copy_official_onnxruntime_dlls TARGET_NAME)
    if(NOT DEFINED ONNXRUNTIME_OFFICIAL_DLL_DIR)
        message(WARNING "ONNXRUNTIME_OFFICIAL_DLL_DIR not set - call fetch_onnxruntime() first")
        return()
    endif()

    if(NOT EXISTS "${ONNXRUNTIME_OFFICIAL_DLL_DIR}/onnxruntime.dll")
        message(WARNING "Official ONNX Runtime DLLs not found at ${ONNXRUNTIME_OFFICIAL_DLL_DIR}")
        return()
    endif()

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_OFFICIAL_DLL_DIR}/onnxruntime.dll"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_OFFICIAL_DLL_DIR}/onnxruntime_providers_shared.dll"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>"
        COMMENT "Copying official ONNX Runtime DLLs for ${TARGET_NAME}"
        VERBATIM
    )
endfunction()
