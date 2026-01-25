
cmake_minimum_required(VERSION 3.15)
# Copyright years for installer and metadata
set(PROJECT_COPYRIGHT_YEARS "2024-2026")

# Prevent vcpkg from copying DLLs (especially incompatible FFmpeg versions) to build output
# We perform explicit deployment using scripts/deploy_tripsitter.ps1 from dedicated directories
if(WIN32)
    set(VCPKG_APPLOCAL_DEPS OFF CACHE BOOL "Disable vcpkg app-local deployment" FORCE)
endif()

project(TripSitterBeatSync)

# Version configuration - can be overridden with -DBEATSYNC_VERSION=...
set(BEATSYNC_VERSION "1.0.0" CACHE STRING "BeatSync library version")
add_definitions(-DBEATSYNC_VERSION="${BEATSYNC_VERSION}")

# TensorRT DLL directory for installer (optional - for RTX GPU acceleration)
# Set via TENSORRT_HOME env var or -DTENSORRT_DLL_DIR=...
if(DEFINED ENV{TENSORRT_HOME})
    set(TENSORRT_DLL_DIR "$ENV{TENSORRT_HOME}/lib" CACHE PATH "TensorRT DLL directory")
else()
    set(TENSORRT_DLL_DIR "C:/TensorRT-10.9.0.34/lib" CACHE PATH "TensorRT DLL directory")
endif()

# AudioFlux DLL directories for installer (optional - for spectral analysis)
set(AUDIOFLUX_DLL_DIR "" CACHE PATH "AudioFlux DLL directory")
set(AUDIOFLUX_FFTW_DLL "" CACHE PATH "AudioFlux FFTW3 DLL path")

# TripSitter.exe path (built separately by Unreal Engine)
# Set via -DTRIPSITTER_EXE_PATH=... or env var
if(DEFINED ENV{TRIPSITTER_EXE_PATH})
    set(TRIPSITTER_EXE_PATH "$ENV{TRIPSITTER_EXE_PATH}" CACHE FILEPATH "Path to TripSitter.exe")
else()
    find_program(TRIPSITTER_EXE_PATH TripSitter.exe
        HINTS
        "C:/UE5_Source/UnrealEngine/Engine/Binaries/Win64"
        DOC "Path to TripSitter.exe"
    )
    if(NOT TRIPSITTER_EXE_PATH)
        message(FATAL_ERROR "TripSitter.exe not found. Please set TRIPSITTER_EXE_PATH variable or export TRIPSITTER_EXE_PATH environment variable to point to the executable.")
    endif()
endif()

# Configure NSIS installer template (run after BEATSYNC_VERSION is set so substitutions work)
configure_file(
    installer/nsis_template.nsi.in
    installer/nsis_template.nsi
    @ONLY
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# FFmpeg setup
# On Windows we default to the dev install path; on other platforms prefer passing
# -DFFMPEG_ROOT to CMake (e.g., use `brew --prefix ffmpeg` on macOS).
if(WIN32)
    set(FFMPEG_ROOT "C:/ffmpeg-dev/ffmpeg-master-latest-win64-gpl-shared")
endif()
if(FFMPEG_ROOT)
    include_directories(${FFMPEG_ROOT}/include)
    link_directories(${FFMPEG_ROOT}/lib)
endif()

# Optional AI / advanced feature flags (stubs included; wire deps when available)
option(USE_ESSENTIA "Enable Essentia-based beat analyzer" OFF)
option(USE_BEATNET "Enable BeatNet Python bridge" OFF)
option(USE_DEMUCS "Enable Demucs stem separation" OFF)
option(USE_GLTRANSITIONS "Enable GLSL transitions library" OFF)
option(USE_AUDIOREACTIVE "Enable FFT-driven audio-reactive effects" OFF)

# Source files
set(CORE_SOURCES
    src/audio/AudioAnalyzer.cpp
    src/audio/BeatGrid.cpp
    src/video/VideoProcessor.cpp
    src/video/VideoWriter.cpp
    src/backend/tracing.cpp
    src/utils/DebugLogger.cpp
)

# Backend library target (wrap core sources into a reusable library and expose a C API)
option(BEATSYNC_BUILD_SHARED "Build beatsync backend as shared library" ON)

# Always provide a static backend for linking by internal targets (beatsync, tests)
add_library(beatsync_backend_static STATIC ${CORE_SOURCES} src/backend/beatsync_capi.cpp)
target_include_directories(beatsync_backend_static PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(beatsync_backend_static PROPERTIES PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/src/backend/beatsync_capi.h")

target_link_libraries(beatsync_backend_static PRIVATE avformat avcodec avutil swresample swscale avfilter)

# Provide a canonical target name 'beatsync_backend' that aliases the static backend for internal linking convenience
add_library(beatsync_backend ALIAS beatsync_backend_static)

# Optionally also build a shared library for external consumers (Unreal plugin)
if(BEATSYNC_BUILD_SHARED AND (WIN32 OR APPLE OR UNIX))
    add_library(beatsync_backend_shared SHARED ${CORE_SOURCES} src/backend/beatsync_capi.cpp)
    # Export C API symbols when building a shared library (Windows uses dllexport; Unix uses visibility attribute)
    target_compile_definitions(beatsync_backend_shared PRIVATE BEATSYNC_CAPI_EXPORT)
    target_include_directories(beatsync_backend_shared PUBLIC ${CMAKE_SOURCE_DIR}/src)
    set_target_properties(beatsync_backend_shared PROPERTIES PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/src/backend/beatsync_capi.h")
    target_link_libraries(beatsync_backend_shared PRIVATE avformat avcodec avutil swresample swscale avfilter)

    # Copy built shared library and headers to unreal-prototype/ThirdParty for Unreal plugin consumption
    add_custom_command(TARGET beatsync_backend_shared POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/include"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/src/backend/beatsync_capi.h" "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/include/"
        # Platform-specific copy destinations
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/lib/x64"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:beatsync_backend_shared>" "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/lib/x64/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_LINKER_FILE:beatsync_backend_shared>" "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/lib/x64/"
        # macOS destination
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/lib/Mac"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:beatsync_backend_shared>" "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/lib/Mac/"
        # Validate that project output does not contain prohibited DLLs (e.g. FFmpeg)
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass -File "${CMAKE_SOURCE_DIR}/scripts/check_no_ffmpeg.ps1" -ReleaseDir "$<TARGET_FILE_DIR:beatsync_backend_shared>"
    )

endif()
# Optional feature sources (always compiled as stubs; actual libs wired via flags)
set(OPTIONAL_SOURCES)

# Tracing support (optional)
option(BEATSYNC_ENABLE_TRACING "Enable OpenTelemetry tracing" OFF)
if(BEATSYNC_ENABLE_TRACING)
    find_package(opentelemetry-cpp CONFIG QUIET)
    if(opentelemetry-cpp_FOUND)
        message(STATUS "OpenTelemetry C++ SDK found - enabling tracing support")
        add_compile_definitions(BEATSYNC_ENABLE_TRACING)
        add_compile_definitions(USE_TRACING)
    else()
        message(WARNING "BEATSYNC_ENABLE_TRACING is ON but opentelemetry-cpp not found. Tracing will be a no-op.")
    endif()
endif()

# Link OpenTelemetry libraries to beatsync after target creation
# (moved to after beatsync target definition)
if(BEATSYNC_ENABLE_TRACING AND opentelemetry-cpp_FOUND)
    target_link_libraries(beatsync_backend_static PRIVATE opentelemetry-cpp::opentelemetry-cpp)
    if(TARGET beatsync_backend_shared)
        target_link_libraries(beatsync_backend_shared PRIVATE opentelemetry-cpp::opentelemetry-cpp)
    endif()
endif()

# BeatNet and Demucs/StemSeparator are now always compiled for AI analysis modes
# These are called via Python scripts and don't need external libraries
list(APPEND OPTIONAL_SOURCES 
    src/audio/BeatNetBridge.cpp
    src/audio/StemSeparator.cpp
)

# TransitionLibrary is lightweight and compiled even when the feature is off
# so the runtime can discover and use shipped shaders when present.
list(APPEND OPTIONAL_SOURCES
    src/video/TransitionLibrary.cpp
)

# ONNX Runtime support for native AI inference (no Python needed)
# Enabled by default - this is the recommended way to use AI beat detection
option(USE_ONNX "Enable ONNX Runtime for native AI inference" ON)
if(USE_ONNX)
    find_package(onnxruntime CONFIG QUIET)
    if(onnxruntime_FOUND)
        message(STATUS "ONNX Runtime found - enabling native AI inference")
        list(APPEND OPTIONAL_SOURCES
            src/audio/OnnxBeatDetector.cpp
            src/audio/OnnxStemSeparator.cpp
            src/audio/OnnxMusicAnalyzer.cpp
        )
        add_definitions(-DUSE_ONNX)

        # Copy ONNX Runtime provider DLLs for GPU support (Windows only)
        if(BEATSYNC_BUILD_SHARED AND WIN32)
            # Use vcpkg-provided variables for portability, fallback to default path
            if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
                set(VCPKG_BIN_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
            else()
                set(VCPKG_BIN_DIR "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin")
            endif()
            # Check for ONNX Runtime provider DLLs and add copy commands conditionally
            set(ONNX_COPY_COMMANDS "")
            
            if(EXISTS "${VCPKG_BIN_DIR}/onnxruntime_providers_shared.dll")
                list(APPEND ONNX_COPY_COMMANDS
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${VCPKG_BIN_DIR}/onnxruntime_providers_shared.dll"
                        "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/lib/x64/"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${VCPKG_BIN_DIR}/onnxruntime_providers_shared.dll"
                        "$<TARGET_FILE_DIR:beatsync_backend_shared>/"
                )
                message(STATUS "ONNX Runtime shared provider DLL found - will copy to output")
            else()
                message(WARNING "onnxruntime_providers_shared.dll not found at ${VCPKG_BIN_DIR} - some ONNX features may not work")
            endif()
            
            if(EXISTS "${VCPKG_BIN_DIR}/onnxruntime_providers_cuda.dll")
                list(APPEND ONNX_COPY_COMMANDS
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${VCPKG_BIN_DIR}/onnxruntime_providers_cuda.dll"
                        "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/lib/x64/"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${VCPKG_BIN_DIR}/onnxruntime_providers_cuda.dll"
                        "$<TARGET_FILE_DIR:beatsync_backend_shared>/"
                )
                message(STATUS "ONNX Runtime CUDA provider DLL found - GPU acceleration available")
            else()
                message(WARNING "onnxruntime_providers_cuda.dll not found at ${VCPKG_BIN_DIR} - GPU acceleration will not be available")
            endif()
            
            if(ONNX_COPY_COMMANDS)
                add_custom_command(TARGET beatsync_backend_shared POST_BUILD
                    ${ONNX_COPY_COMMANDS}
                    COMMENT "Copying ONNX Runtime GPU provider DLLs"
                )
            endif()

            # Copy TensorRT DLLs for RTX acceleration (optional - skip if not found)
            if(EXISTS "${TENSORRT_DLL_DIR}/nvinfer_10.dll")
                add_custom_command(TARGET beatsync_backend_shared POST_BUILD
                    # Copy to ThirdParty for Unreal
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${TENSORRT_DLL_DIR}/nvinfer_10.dll"
                        "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/lib/x64/"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${TENSORRT_DLL_DIR}/nvinfer_lean_10.dll"
                        "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/lib/x64/"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${TENSORRT_DLL_DIR}/nvinfer_plugin_10.dll"
                        "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/lib/x64/"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${TENSORRT_DLL_DIR}/nvinfer_dispatch_10.dll"
                        "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/lib/x64/"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${TENSORRT_DLL_DIR}/nvonnxparser_10.dll"
                        "${CMAKE_SOURCE_DIR}/unreal-prototype/ThirdParty/beatsync/lib/x64/"
                    # Copy to build output directory for direct testing
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${TENSORRT_DLL_DIR}/nvinfer_10.dll"
                        "$<TARGET_FILE_DIR:beatsync_backend_shared>/"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${TENSORRT_DLL_DIR}/nvinfer_lean_10.dll"
                        "$<TARGET_FILE_DIR:beatsync_backend_shared>/"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${TENSORRT_DLL_DIR}/nvinfer_plugin_10.dll"
                        "$<TARGET_FILE_DIR:beatsync_backend_shared>/"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${TENSORRT_DLL_DIR}/nvinfer_dispatch_10.dll"
                        "$<TARGET_FILE_DIR:beatsync_backend_shared>/"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${TENSORRT_DLL_DIR}/nvonnxparser_10.dll"
                        "$<TARGET_FILE_DIR:beatsync_backend_shared>/"
                    COMMENT "Copying TensorRT DLLs for RTX acceleration"
                )
            else()
                message(STATUS "TensorRT DLLs not found at ${TENSORRT_DLL_DIR} - TensorRT acceleration disabled")
            endif()
        endif()
    else()
        message(WARNING "ONNX Runtime not found. Install with: vcpkg install onnxruntime-gpu")
    endif()
endif()

# AudioFlux support for signal processing-based beat detection (no Python, no GPU needed)
# This provides reliable spectral flux onset detection as an alternative to neural networks
option(USE_AUDIOFLUX "Enable AudioFlux for spectral analysis" ON)
set(AUDIOFLUX_ROOT "" CACHE PATH "AudioFlux installation directory (optional, overrides autodetect)")
if(USE_AUDIOFLUX)
    # Try to find AudioFlux include dir
    find_path(AUDIOFLUX_INCLUDE_DIR flux_base.h
        PATHS
            "${AUDIOFLUX_ROOT}/include"
            /usr/local/include
            /usr/include
            $ENV{AUDIOFLUX_ROOT}/include
        DOC "AudioFlux include directory"
    )
    # Try to find AudioFlux library
    find_library(AUDIOFLUX_LIBRARY audioflux
        PATHS
            "${AUDIOFLUX_ROOT}/build/windowBuild/Release"
            "${AUDIOFLUX_ROOT}/lib"
            /usr/local/lib
            /usr/lib
            $ENV{AUDIOFLUX_ROOT}/lib
        DOC "AudioFlux library"
    )
    if(AUDIOFLUX_INCLUDE_DIR AND AUDIOFLUX_LIBRARY)
        message(STATUS "AudioFlux found: ${AUDIOFLUX_INCLUDE_DIR}, ${AUDIOFLUX_LIBRARY} - enabling spectral analysis")
        list(APPEND OPTIONAL_SOURCES
            src/audio/AudioFluxBeatDetector.cpp
        )
        add_definitions(-DUSE_AUDIOFLUX)

        # Set AudioFlux DLL paths for installer
        get_filename_component(AUDIOFLUX_LIB_DIR_FOR_INSTALLER "${AUDIOFLUX_LIBRARY}" DIRECTORY)
        set(AUDIOFLUX_DLL_DIR "${AUDIOFLUX_LIB_DIR_FOR_INSTALLER}" CACHE PATH "AudioFlux DLL directory" FORCE)
        if(AUDIOFLUX_ROOT)
            set(AUDIOFLUX_FFTW_DLL "${AUDIOFLUX_ROOT}/python/audioflux/lib/libfftw3f-3.dll" CACHE FILEPATH "AudioFlux FFTW3 DLL path" FORCE)
        endif()
    else()
        message(WARNING "AudioFlux not found (include: ${AUDIOFLUX_INCLUDE_DIR}, lib: ${AUDIOFLUX_LIBRARY}). Spectral analysis disabled.")
    endif()
endif()

# Bandlimited resampler support (libsamplerate)
find_package(SampleRate CONFIG QUIET)
if(SampleRate_FOUND)
    message(STATUS "libsamplerate found - enabling high-quality resampling")
    add_definitions(-DUSE_LIBSAMPLERATE)
endif()

if(USE_ESSENTIA)
    list(APPEND OPTIONAL_SOURCES src/audio/EssentiaAnalyzer.cpp)
    add_definitions(-DUSE_ESSENTIA)
endif()
if(USE_GLTRANSITIONS)
    list(APPEND OPTIONAL_SOURCES src/video/TransitionLibrary.cpp)
    add_definitions(-DUSE_GLTRANSITIONS)
endif()
if(USE_AUDIOREACTIVE)
    list(APPEND OPTIONAL_SOURCES src/audio/FFTAnalyzer.cpp src/video/AudioReactiveEffects.cpp)
    add_definitions(-DUSE_AUDIOREACTIVE)
endif()

# Ensure optional sources are added to the backend library targets
if(TARGET beatsync_backend_static)
    target_sources(beatsync_backend_static PRIVATE ${OPTIONAL_SOURCES})
endif()
if(TARGET beatsync_backend_shared)
    target_sources(beatsync_backend_shared PRIVATE ${OPTIONAL_SOURCES})
endif()

# CLI executable
add_executable(beatsync src/main.cpp)
target_link_libraries(beatsync PRIVATE
    beatsync_backend_static
    avformat avcodec avutil swresample swscale avfilter
)
if(USE_ONNX AND onnxruntime_FOUND)
    target_link_libraries(beatsync PRIVATE onnxruntime::onnxruntime)
    target_link_libraries(beatsync_backend_static PRIVATE onnxruntime::onnxruntime)
    if(TARGET beatsync_backend_shared)
        target_link_libraries(beatsync_backend_shared PRIVATE onnxruntime::onnxruntime)
    endif()
endif()

# AudioFlux linking (signal processing-based beat detection)
if(USE_AUDIOFLUX AND EXISTS "${AUDIOFLUX_LIBRARY}")
    target_include_directories(beatsync PRIVATE "${AUDIOFLUX_INCLUDE_DIR}")
    target_include_directories(beatsync_backend_static PRIVATE "${AUDIOFLUX_INCLUDE_DIR}")
    target_link_libraries(beatsync PRIVATE "${AUDIOFLUX_LIBRARY}")
    target_link_libraries(beatsync_backend_static PRIVATE "${AUDIOFLUX_LIBRARY}")
    if(TARGET beatsync_backend_shared)
        target_include_directories(beatsync_backend_shared PRIVATE "${AUDIOFLUX_INCLUDE_DIR}")
        target_link_libraries(beatsync_backend_shared PRIVATE "${AUDIOFLUX_LIBRARY}")
        # Copy AudioFlux DLLs to output directory (Windows only)
        # Use cache variables AUDIOFLUX_DLL_DIR and AUDIOFLUX_FFTW_DLL for consistency with installer
        if(WIN32 AND AUDIOFLUX_DLL_DIR)
            set(AUDIOFLUX_DLL_PATH "${AUDIOFLUX_DLL_DIR}/audioflux.dll")
            if(EXISTS "${AUDIOFLUX_DLL_PATH}" AND EXISTS "${AUDIOFLUX_FFTW_DLL}")
                add_custom_command(TARGET beatsync_backend_shared POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${AUDIOFLUX_DLL_PATH}"
                        "$<TARGET_FILE_DIR:beatsync_backend_shared>/"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${AUDIOFLUX_FFTW_DLL}"
                        "$<TARGET_FILE_DIR:beatsync_backend_shared>/"
                    COMMENT "Copying AudioFlux DLLs"
                )
            else()
                message(WARNING "AudioFlux DLLs not found: ${AUDIOFLUX_DLL_PATH} or ${AUDIOFLUX_FFTW_DLL}")
            endif()
        endif()
    endif()
endif()

# libsamplerate linking (high-quality resampling)
if(SampleRate_FOUND)
    target_link_libraries(beatsync PRIVATE SampleRate::samplerate)
    target_link_libraries(beatsync_backend_static PRIVATE SampleRate::samplerate)
    if(TARGET beatsync_backend_shared)
        target_link_libraries(beatsync_backend_shared PRIVATE SampleRate::samplerate)
    endif()
endif()

# Output directory for beatsync (always available)
set_target_properties(beatsync PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)

# =============================================================================
# INSTALLATION FROM PRE-PACKAGED UE BUILD
# =============================================================================
# Use the pre-packaged Unreal Engine build which includes all cooked content,
# shaders, and engine dependencies. This is more reliable than piecing together
# files from the UE source tree.

if(DEFINED ENV{UE_PACKAGED_BUILD})
    set(UE_PACKAGED_BUILD "$ENV{UE_PACKAGED_BUILD}" CACHE PATH "Path to packaged build")
else()
    set(UE_PACKAGED_BUILD "" CACHE PATH "Path to packaged build")
endif()

if(UE_PACKAGED_BUILD AND EXISTS "${UE_PACKAGED_BUILD}/MyProject.exe")
    message(STATUS "Pre-packaged UE build found at ${UE_PACKAGED_BUILD}")

    # Install root launcher
    install(PROGRAMS "${UE_PACKAGED_BUILD}/MyProject.exe"
        DESTINATION "."
        RENAME "TripSitter.exe"
    )

    # Install manifest files
    install(FILES
        "${UE_PACKAGED_BUILD}/Manifest_DebugFiles_Win64.txt"
        "${UE_PACKAGED_BUILD}/Manifest_NonUFSFiles_Win64.txt"
        "${UE_PACKAGED_BUILD}/Manifest_UFSFiles_Win64.txt"
        DESTINATION "."
        OPTIONAL
    )

    # Install Engine folder (Binaries/ThirdParty, Config, Content, Extras, Plugins)
    install(DIRECTORY "${UE_PACKAGED_BUILD}/Engine/" DESTINATION "Engine")

    # Copy Slate/Common from UE source (packaged build only has Cursor, missing UI textures)
    set(UE_SLATE_DIR "C:/UE5_Source/UnrealEngine/Engine/Content/Slate")
    if(EXISTS "${UE_SLATE_DIR}/Common")
        install(DIRECTORY "${UE_SLATE_DIR}/Common" DESTINATION "Engine/Content/Slate")
    endif()

    # Install MyProject folder (renamed to TripSitter)
    # This contains the cooked game content and binaries
    install(DIRECTORY "${UE_PACKAGED_BUILD}/MyProject/" DESTINATION "TripSitter")

    # Install updated beatsync_backend_shared.dll (overwrite packaged version)
    if(TARGET beatsync_backend_shared)
        install(TARGETS beatsync_backend_shared
            RUNTIME DESTINATION "TripSitter/Binaries/Win64"
            LIBRARY DESTINATION "TripSitter/Binaries/Win64"
        )
    endif()

    # Install AudioFlux DLLs if available (may not be in packaged build)
    if(USE_AUDIOFLUX AND AUDIOFLUX_DLL_DIR)
        if(EXISTS "${AUDIOFLUX_DLL_DIR}/audioflux.dll")
            install(FILES "${AUDIOFLUX_DLL_DIR}/audioflux.dll"
                DESTINATION "TripSitter/Binaries/Win64" OPTIONAL)
        endif()
        if(EXISTS "${AUDIOFLUX_FFTW_DLL}")
            install(FILES "${AUDIOFLUX_FFTW_DLL}"
                DESTINATION "TripSitter/Binaries/Win64" OPTIONAL)
        endif()
    endif()

    # Install UI resources next to game executable (where widget looks for them)
    set(RESOURCE_SRC_DIR "${CMAKE_SOURCE_DIR}/unreal-prototype/Source/TripSitter/Resources")
    if(EXISTS "${RESOURCE_SRC_DIR}")
        install(FILES
            "${RESOURCE_SRC_DIR}/wallpaper.png"
            "${RESOURCE_SRC_DIR}/TitleHeader.png"
            "${RESOURCE_SRC_DIR}/Corpta.otf"
            DESTINATION "TripSitter/Binaries/Win64/Resources"
            OPTIONAL
        )
    endif()

else()
    message(WARNING "Pre-packaged UE build not found at ${UE_PACKAGED_BUILD}")
    message(WARNING "Falling back to legacy installation method")

    # Legacy installation (may not work properly)
    set(UE_BIN_DIR "Engine/Binaries/Win64")

    if(EXISTS "${TRIPSITTER_EXE_PATH}")
        install(PROGRAMS "${TRIPSITTER_EXE_PATH}" DESTINATION "${UE_BIN_DIR}")
    endif()

    if(TARGET beatsync_backend_shared)
        install(TARGETS beatsync_backend_shared
            RUNTIME DESTINATION "${UE_BIN_DIR}"
            LIBRARY DESTINATION "${UE_BIN_DIR}"
        )
    endif()
endif()

# Install ONNX models if they exist
if(EXISTS "${CMAKE_SOURCE_DIR}/models")
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/models" DESTINATION .)
endif()

# Install license files
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE" DESTINATION licenses)
if(EXISTS "${CMAKE_SOURCE_DIR}/THIRD_PARTY_LICENSES.md")
    install(FILES "${CMAKE_SOURCE_DIR}/THIRD_PARTY_LICENSES.md" DESTINATION licenses)
endif()

# Install changelog
if(EXISTS "${CMAKE_SOURCE_DIR}/CHANGELOG.md")
    install(FILES "${CMAKE_SOURCE_DIR}/CHANGELOG.md" DESTINATION .)
endif()

# Also install the Python scripts as fallback (for users who have Python)
install(FILES
    "${CMAKE_SOURCE_DIR}/scripts/beatnet_analyze.py"
    "${CMAKE_SOURCE_DIR}/scripts/demucs_separate.py"
    DESTINATION scripts
    OPTIONAL
)

# CPack packaging configuration (ZIP + NSIS for Windows)
include(InstallRequiredSystemLibraries)
set(CPACK_PACKAGE_NAME "MTVTripSitter")
set(CPACK_PACKAGE_VENDOR "MTV")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "MTV Trip Sitter - Audio Beat Sync")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "MTV TripSitter")

# Platform-specific CPack generators
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")

    # NSIS configuration for professional Windows installer
    set(CPACK_NSIS_DISPLAY_NAME "MTV Trip Sitter")
    set(CPACK_NSIS_PACKAGE_NAME "MTV Trip Sitter")
    set(CPACK_NSIS_CONTACT "samuelharnischmacher@gmail.com")
    set(CPACK_NSIS_HELP_LINK "https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI")
    set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI")

    # Icon for installer
    if(EXISTS "${CMAKE_SOURCE_DIR}/assets/icon.ico")
        set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/assets/icon.ico")
        set(CPACK_NSIS_MUI_UNIICON "${CMAKE_SOURCE_DIR}/assets/icon.ico")
        set(CPACK_NSIS_INSTALLED_ICON_NAME "TripSitter.exe")
    endif()

    # Start menu and desktop shortcuts
    set(CPACK_NSIS_CREATE_ICONS_EXTRA
        "CreateShortCut '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\MTV TripSitter.lnk' '$INSTDIR\\\\TripSitter.exe'"
    )
    set(CPACK_NSIS_DELETE_ICONS_EXTRA
        "Delete '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\MTV TripSitter.lnk'"
    )

    # Add option for desktop shortcut
    set(CPACK_NSIS_MODIFY_PATH OFF)
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)

    # Request admin rights for installation
    set(CPACK_NSIS_EXECUTABLES_DIRECTORY ".")

    # Install to Program Files by default
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")

elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "MTV Trip Sitter")
    set(CPACK_DMG_FORMAT "UDZO")
else()
    set(CPACK_GENERATOR "TGZ")
endif()

# Tests (disabled by default). Enable with -DBUILD_TESTS=ON
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

include(CPack)
