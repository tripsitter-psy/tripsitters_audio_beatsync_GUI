cmake_minimum_required(VERSION 3.15)
project(TripSitterBeatSync)

# Version configuration - can be overridden with -DBEATSYNC_VERSION=...
set(BEATSYNC_VERSION "1.0.0" CACHE STRING "BeatSync library version")
add_definitions(-DBEATSYNC_VERSION="${BEATSYNC_VERSION}")

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
    )
endif()
# Optional feature sources (always compiled as stubs; actual libs wired via flags)
set(OPTIONAL_SOURCES)

# Tracing support (optional)
option(BEATSYNC_ENABLE_TRACING "Enable OpenTelemetry tracing" OFF)
if(BEATSYNC_ENABLE_TRACING)
    # Try to discover OpenTelemetry C++ SDK via CMake config package (vcpkg provides opentelemetry-cpp)
    find_package(opentelemetry-cpp CONFIG QUIET)
    if(opentelemetry-cpp_FOUND)
        message(STATUS "OpenTelemetry C++ SDK found - enabling tracing support")
        add_compile_definitions(BEATSYNC_ENABLE_TRACING)
        add_compile_definitions(USE_TRACING)
        # Link OpenTelemetry libraries to beatsync_backend_shared
        if(TARGET beatsync_backend_shared)
            target_link_libraries(beatsync_backend_shared PRIVATE opentelemetry_cpp_trace opentelemetry_cpp_sdk)
        endif()
    else()
        message(WARNING "BEATSYNC_ENABLE_TRACING is ON but opentelemetry-cpp not found. Tracing will be a no-op.")
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
    else()
        message(WARNING "ONNX Runtime not found. Install with: vcpkg install onnxruntime-gpu")
    endif()
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

# Output directory for beatsync (always available)
set_target_properties(beatsync PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)

# Install AI tools if they exist (bundled executables for beat detection)
# These are built separately with PyInstaller - see scripts/build_ai_tools.py
if(EXISTS "${CMAKE_SOURCE_DIR}/scripts/dist/beatnet_analyze.exe")
    install(PROGRAMS "${CMAKE_SOURCE_DIR}/scripts/dist/beatnet_analyze.exe" DESTINATION ai_tools)
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/scripts/dist/demucs_separate.exe")
    install(PROGRAMS "${CMAKE_SOURCE_DIR}/scripts/dist/demucs_separate.exe" DESTINATION ai_tools)
endif()


# Install ONNX models if they exist (preferred method - no Python needed)
if(EXISTS "${CMAKE_SOURCE_DIR}/models")
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/models" DESTINATION .)
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
        set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\beatsync.exe")
    endif()

    # Start menu and desktop shortcuts
    set(CPACK_NSIS_CREATE_ICONS_EXTRA
        "CreateShortCut '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\MTV TripSitter.lnk' '$INSTDIR\\\\bin\\\\beatsync.exe'"
    )
    set(CPACK_NSIS_DELETE_ICONS_EXTRA
        "Delete '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\MTV TripSitter.lnk'"
    )

    # Add option for desktop shortcut
    set(CPACK_NSIS_MODIFY_PATH OFF)
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)

    # Request admin rights for installation
    set(CPACK_NSIS_EXECUTABLES_DIRECTORY "bin")

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
