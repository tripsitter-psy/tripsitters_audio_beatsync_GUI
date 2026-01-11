# Tests for BeatSyncEditor

# =============================================================================
# Official ONNX Runtime Download (Workaround for vcpkg build issues)
# =============================================================================
# vcpkg's ONNX Runtime 1.23.2 has a heap corruption bug on Windows.
# We download official Microsoft binaries instead.
# See: https://github.com/microsoft/vcpkg/issues/49349
#
include(${CMAKE_SOURCE_DIR}/cmake/FetchOnnxRuntime.cmake)

if(WIN32)
    # Download official ONNX Runtime binaries
    # OPTIONAL: set the SHA256 for the official archive to enable verification.
    # You can compute this locally (sha256sum on Linux) and paste the value here.
    # Example: set(ONNXRUNTIME_OFFICIAL_SHA256 "<64-hex-sha256>")
    if(NOT DEFINED ONNXRUNTIME_OFFICIAL_SHA256)
        set(ONNXRUNTIME_OFFICIAL_SHA256 "")
    endif()

    fetch_onnxruntime(VERSION 1.23.2 DESTINATION "${CMAKE_BINARY_DIR}/onnxruntime-official" SHA256 ${ONNXRUNTIME_OFFICIAL_SHA256})

    # Fallback to vcpkg if official download fails
    set(VCPKG_BIN_DIR "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin")
endif()

# =============================================================================
# DLL Copy Helper for Windows Tests
# =============================================================================
# On Windows, tests that link to ONNX Runtime or FFmpeg need runtime DLLs.
# This helper copies required DLLs to the test output directory.
#
# CUDA Support:
#   - By default, only CPU DLLs are copied (avoids crashes if CUDA isn't installed)
#   - Set BEATSYNC_ONNX_USE_CUDA=1 environment variable at runtime to use CUDA
#   - For CUDA tests, also copy onnxruntime_providers_cuda.dll manually or via CI
#
if(WIN32)
    # Function to copy runtime DLLs for a test target
    # Uses official ONNX Runtime binaries (workaround for vcpkg bug)
    # Usage: copy_runtime_dlls_for_test(target_name [INCLUDE_CUDA])
    function(copy_runtime_dlls_for_test TARGET_NAME)
        cmake_parse_arguments(ARG "INCLUDE_CUDA" "" "" ${ARGN})

        # Prefer official ONNX Runtime binaries over vcpkg (vcpkg has heap corruption bug)
        if(DEFINED ONNXRUNTIME_OFFICIAL_DLL_DIR AND EXISTS "${ONNXRUNTIME_OFFICIAL_DLL_DIR}/onnxruntime.dll")
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
        elseif(EXISTS "${VCPKG_BIN_DIR}/onnxruntime.dll")
            # Fallback to vcpkg (may have issues)
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${VCPKG_BIN_DIR}/onnxruntime.dll"
                    "$<TARGET_FILE_DIR:${TARGET_NAME}>"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${VCPKG_BIN_DIR}/onnxruntime_providers_shared.dll"
                    "$<TARGET_FILE_DIR:${TARGET_NAME}>"
                COMMENT "Copying ONNX Runtime DLLs for ${TARGET_NAME} (vcpkg fallback)"
                VERBATIM
            )
        endif()

        # Optionally copy CUDA provider DLL (335+ MB, only for GPU testing)
        # CUDA provider is only available from vcpkg build
        if(ARG_INCLUDE_CUDA AND EXISTS "${VCPKG_BIN_DIR}/onnxruntime_providers_cuda.dll")
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${VCPKG_BIN_DIR}/onnxruntime_providers_cuda.dll"
                    "$<TARGET_FILE_DIR:${TARGET_NAME}>"
                COMMENT "Copying ONNX Runtime CUDA provider for ${TARGET_NAME}"
                VERBATIM
            )
        endif()
    endfunction()
endif()

# Simple test that verifies the ZIP created by BeatSync::createZip uses the "store" method
# until DEFLATE is implemented. This test is intentionally standalone and doesn't require
# a test framework; it will return non-zero on failure so CTest will mark it failed.

add_executable(test_logarchiver_deflate
    test_logarchiver_deflate.cpp
    ../src/utils/LogArchiver.cpp
)

target_include_directories(test_logarchiver_deflate PRIVATE ${CMAKE_SOURCE_DIR}/src/utils)
add_test(NAME logarchiver_deflate COMMAND test_logarchiver_deflate)

if(BUILD_TESTS)
    # Catch2-based stub test (enabled only when BUILD_TESTS=ON). This provides a minimal
    # unit-test scaffold so contributors can add assertions when DEFLATE is implemented.
    include(FetchContent)
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(catch2)

    add_executable(test_logarchiver_catch2
        catch_main.cpp
        test_deflate_catch2.cpp
        ../src/utils/LogArchiver.cpp
    )

    target_include_directories(test_logarchiver_catch2 PRIVATE ${CMAKE_SOURCE_DIR}/src/utils)
    target_link_libraries(test_logarchiver_catch2 PRIVATE Catch2::Catch2)
    add_test(NAME logarchiver_catch2 COMMAND test_logarchiver_catch2)
endif()

# Transition library tests
add_executable(test_transition_library
    catch_main.cpp
    test_transition_library.cpp
    ../src/video/TransitionLibrary.cpp
)

target_include_directories(test_transition_library PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_transition_library PRIVATE Catch2::Catch2)
add_test(NAME transition_library COMMAND test_transition_library)
set_tests_properties(transition_library PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# FFTAnalyzer test
add_executable(test_fft_analyzer
    catch_main.cpp
    test_fft_analyzer.cpp
    ../src/audio/FFTAnalyzer.cpp
)

target_include_directories(test_fft_analyzer PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_fft_analyzer PRIVATE Catch2::Catch2)
add_test(NAME fft_analyzer COMMAND test_fft_analyzer)
set_tests_properties(fft_analyzer PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# BeatNetBridge test
add_executable(test_beatnet_bridge
    catch_main.cpp
    test_beatnet_bridge.cpp
    ../src/audio/BeatNetBridge.cpp
    ../src/audio/BeatGrid.cpp
)

target_include_directories(test_beatnet_bridge PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_beatnet_bridge PRIVATE Catch2::Catch2)
add_test(NAME beatnet_bridge COMMAND test_beatnet_bridge)
set_tests_properties(beatnet_bridge PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# BeatNet Python integration (gated by env var BEATSYNC_ENABLE_PYTHON)
add_executable(test_beatnet_python
    catch_main.cpp
    test_beatnet_python.cpp
    ../src/audio/BeatNetBridge.cpp
    ../src/audio/BeatGrid.cpp
)

target_include_directories(test_beatnet_python PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_beatnet_python PRIVATE Catch2::Catch2)
add_test(NAME beatnet_python COMMAND test_beatnet_python)
set_tests_properties(beatnet_python PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# OnnxDetector prototype test
add_executable(test_onnx_detector
    catch_main.cpp
    test_onnx_detector.cpp
    ../src/audio/BeatGrid.cpp
)

target_include_directories(test_onnx_detector PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_onnx_detector PRIVATE Catch2::Catch2 beatsync_backend_static)
add_test(NAME onnx_detector COMMAND test_onnx_detector)
set_tests_properties(onnx_detector PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Regression: repeated ONNX inference to catch allocator / heap corruption regressions
add_executable(test_onnx_detector_regression
    catch_main.cpp
    test_onnx_detector_regression.cpp
    ../src/audio/BeatGrid.cpp
)

target_include_directories(test_onnx_detector_regression PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_onnx_detector_regression PRIVATE Catch2::Catch2 beatsync_backend_static)
add_test(NAME onnx_detector_regression COMMAND test_onnx_detector_regression)
set_tests_properties(onnx_detector_regression PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Helper-process ONNX inference test (runs in isolated process to avoid crashing test harness)
add_executable(test_onnx_inference_helper
    onnx_inference_helper.cpp
    ../src/audio/BeatGrid.cpp
)

target_include_directories(test_onnx_inference_helper PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_onnx_inference_helper PRIVATE Catch2::Catch2 beatsync_backend_static)
add_test(NAME onnx_inference_helper COMMAND test_onnx_inference_helper)
set_tests_properties(onnx_inference_helper PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# On Windows, copy ONNX Runtime DLLs to test output directory for regression tests as well
if(WIN32)
    copy_runtime_dlls_for_test(test_onnx_detector_regression)
    copy_runtime_dlls_for_test(test_onnx_detector_regression INCLUDE_CUDA)
    copy_runtime_dlls_for_test(test_onnx_inference_helper)
    copy_runtime_dlls_for_test(test_onnx_inference_helper INCLUDE_CUDA)
    set_tests_properties(onnx_detector_regression PROPERTIES
        ENVIRONMENT "PATH=${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin;$ENV{PATH}"
    )
    set_tests_properties(onnx_inference_helper PROPERTIES
        ENVIRONMENT "PATH=${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin;$ENV{PATH}"
    )
endif()

# ONNX tests now use official Microsoft binaries (workaround for vcpkg heap corruption)
# See: https://github.com/microsoft/vcpkg/issues/49349
set_tests_properties(onnx_inference_helper onnx_detector_regression onnx_detector PROPERTIES
    LABELS "onnx"
)


# On Windows, copy ONNX Runtime DLLs to test output directory
if(WIN32)
    copy_runtime_dlls_for_test(test_onnx_detector)
    # Also prepare optional CUDA provider copy (only copied if the provider DLL exists in vcpkg)
    copy_runtime_dlls_for_test(test_onnx_detector INCLUDE_CUDA)
    set_tests_properties(onnx_detector PROPERTIES
        ENVIRONMENT "PATH=${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin;$ENV{PATH}"
    )
endif()

# CUDA-enabled ONNX test (same executable, different test config with CUDA env var)
# This test is only useful when CUDA runtime is installed and GPU is available
# Run with: ctest -R onnx_detector_cuda
add_test(NAME onnx_detector_cuda COMMAND test_onnx_detector)
set_tests_properties(onnx_detector_cuda PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    LABELS "cuda;gpu"
)
if(WIN32)
    # For CUDA test, set env var to enable CUDA provider
    # Also copy CUDA provider DLL into the test dir (if present)
    copy_runtime_dlls_for_test(test_onnx_detector INCLUDE_CUDA)
    set_tests_properties(onnx_detector_cuda PROPERTIES
        ENVIRONMENT "BEATSYNC_ONNX_USE_CUDA=1;PATH=${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin;$ENV{PATH}"
    )
endif()

# CUDA test uses official binaries for CPU portion; CUDA provider still from vcpkg if available
set_tests_properties(onnx_detector_cuda PROPERTIES
    LABELS "onnx;cuda;gpu"
)

# SpectralFlux tests
add_executable(test_spectral_flux
    catch_main.cpp
    test_spectral_flux.cpp
    ../src/audio/SpectralFlux.cpp
)

target_include_directories(test_spectral_flux PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_spectral_flux PRIVATE Catch2::Catch2)
add_test(NAME spectral_flux COMMAND test_spectral_flux)
set_tests_properties(spectral_flux PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Tracing tests (requires USE_TRACING=ON to build/run)
if(USE_TRACING)
    add_executable(test_tracing
        catch_main.cpp
        test_tracing.cpp
        ../src/tracing/Tracing.cpp
    )
    target_include_directories(test_tracing PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_tracing PRIVATE Catch2::Catch2)
    add_test(NAME tracing COMMAND test_tracing)
    set_tests_properties(tracing PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

    # Provide a small static tracing library to satisfy TRACE_FUNC/TRACE_SCOPE references
    add_library(beatsync_tracing STATIC ${CMAKE_SOURCE_DIR}/src/tracing/Tracing.cpp)
    target_include_directories(beatsync_tracing PUBLIC ${CMAKE_SOURCE_DIR}/src)

    # Link the tracing library into test executables that use the TRACE_* macros
    if(TARGET test_spectral_flux)
        target_link_libraries(test_spectral_flux PRIVATE beatsync_tracing)
    endif()
    if(TARGET test_onnx_detector)
        target_link_libraries(test_onnx_detector PRIVATE beatsync_tracing)
    endif()
    if(TARGET test_onnx_detector_regression)
        target_link_libraries(test_onnx_detector_regression PRIVATE beatsync_tracing)
    endif()
endif()

# Transition chaining tests - disabled: tests private method
# add_executable(test_transition_chaining
#     catch_main.cpp
#     test_transition_chaining.cpp
#     ../src/video/VideoWriter.cpp
#     ../src/video/TransitionLibrary.cpp
# )
# target_include_directories(test_transition_chaining PRIVATE ${CMAKE_SOURCE_DIR}/src)
# target_link_libraries(test_transition_chaining PRIVATE Catch2::Catch2)
# add_test(NAME transition_chaining COMMAND test_transition_chaining)
# set_tests_properties(transition_chaining PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Backend API smoke test
add_executable(test_backend_api
    catch_main.cpp
    test_backend_api.cpp
)

target_include_directories(test_backend_api PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/backend)
target_link_libraries(test_backend_api PRIVATE Catch2::Catch2 beatsync_backend_static avformat avcodec avutil swresample swscale avfilter)
add_test(NAME backend_api COMMAND test_backend_api)
set_tests_properties(backend_api PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# On Windows, copy DLLs and set PATH for the test
if(WIN32)
    copy_runtime_dlls_for_test(test_backend_api)
    copy_runtime_dlls_for_test(test_backend_api INCLUDE_CUDA)
    set_tests_properties(backend_api PROPERTIES
        ENVIRONMENT "PATH=${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin;$ENV{PATH}"
    )
endif()

# Verify wallpaper is packaged into build outputs or app bundle
if(BUILD_GUI)
    add_executable(test_wallpaper_installed
        test_wallpaper_installed.cpp
    )

    target_link_libraries(test_wallpaper_installed PRIVATE Catch2::Catch2)
    # Pass two candidate locations: non-mac build assets path and macOS bundle assets path
    if(APPLE)
        set(MAC_WALLPAPER_PATH "$<TARGET_BUNDLE_CONTENT_DIR:TripSitter>/MacOS/assets/wallpaper.png")
    else()
        # On non-Apple platforms the bundle path doesn't exist - pass the non-mac build path twice so the test still receives two arguments
        set(MAC_WALLPAPER_PATH "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/assets/wallpaper.png")
    endif()
    add_test(NAME wallpaper_installed COMMAND test_wallpaper_installed "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/assets/wallpaper.png" "${MAC_WALLPAPER_PATH}")
    set_tests_properties(wallpaper_installed PROPERTIES WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

# Runtime check: launch the GUI executable with --check-wallpaper and assert success when wallpaper found
if(BUILD_GUI)
    add_test(NAME wallpaper_runtime_check COMMAND $<TARGET_FILE:TripSitter> --check-wallpaper)
    set_tests_properties(wallpaper_runtime_check PROPERTIES WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
