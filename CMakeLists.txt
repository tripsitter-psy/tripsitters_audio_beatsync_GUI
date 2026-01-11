cmake_minimum_required(VERSION 3.15)
project(TripSitterBeatSync)

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

# (wxWidgets support removed)
# On Windows we ship a default path; on macOS/Linux we let find_package find it
# wxWidgets support removed (wxUniversal option removed since wxWidgets is no longer used)

# Optional AI / advanced feature flags (stubs included; wire deps when available)
option(USE_ESSENTIA "Enable Essentia-based beat analyzer" OFF)
option(USE_BEATNET "Enable BeatNet Python bridge" OFF)
# Enable invoking Python subprocess for BeatNet (dev-only; DO NOT enable for release builds)
option(ENABLE_BEATNET_PYTHON "Enable invoking Python subprocess for BeatNet (dev only)" OFF)
if(ENABLE_BEATNET_PYTHON)
    add_compile_definitions(ENABLE_BEATNET_PYTHON)
endif()
option(USE_DEMUCS "Enable Demucs stem separation" OFF)
option(USE_GLTRANSITIONS "Enable GLSL transitions library" OFF)
option(USE_AUDIOREACTIVE "Enable FFT-driven audio-reactive effects" OFF)

# BeatNet Python integration - OFF by default for packaged builds
# Enable with -DENABLE_BEATNET_PYTHON=ON for development only
# Runtime can also be enabled via BEATSYNC_ENABLE_PYTHON=1 environment variable
option(ENABLE_BEATNET_PYTHON "Enable BeatNet Python subprocess invocation (dev only)" OFF)
if(ENABLE_BEATNET_PYTHON)
    add_definitions(-DENABLE_BEATNET_PYTHON)
    message(STATUS "BeatNet Python subprocess ENABLED - this should NOT be used in release builds")
endif()

# Build options
option(BUILD_GUI "Build TripSitter GUI" ON)

# If GUI build is requested, check for wxWidgets; if not found, disable GUI early
if(BUILD_GUI)
    find_package(wxWidgets QUIET COMPONENTS core base adv)
    if(NOT wxWidgets_FOUND)
        message(STATUS "wxWidgets not found; disabling TripSitter GUI build")
        set(BUILD_GUI OFF)
    endif()
endif()


# wxWidgets find_package removed (no longer used)

# Source files
set(CORE_SOURCES
    src/audio/AudioAnalyzer.cpp
    src/audio/BeatGrid.cpp
    src/video/VideoProcessor.cpp
    src/video/VideoWriter.cpp
)

# Backend library target (wrap core sources into a reusable library and expose a C API)
option(BEATSYNC_BUILD_SHARED "Build beatsync backend as shared library" ON)

# Always provide a static backend for linking by internal targets (TripSitter, beatsync, tests)
add_library(beatsync_backend_static STATIC ${CORE_SOURCES} ${OPTIONAL_SOURCES} src/backend/beatsync_capi.cpp)
target_include_directories(beatsync_backend_static PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(beatsync_backend_static PROPERTIES PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/src/backend/beatsync_capi.h")

target_link_libraries(beatsync_backend_static PRIVATE avformat avcodec avutil swresample swscale avfilter)

# Provide a canonical target name 'beatsync_backend' that aliases the static backend for internal linking convenience
add_library(beatsync_backend ALIAS beatsync_backend_static)

# Optionally also build a shared library for external consumers (Unreal plugin)
if(BEATSYNC_BUILD_SHARED AND (WIN32 OR APPLE OR UNIX))
    add_library(beatsync_backend_shared SHARED ${CORE_SOURCES} ${OPTIONAL_SOURCES} src/backend/beatsync_capi.cpp)
    # Export C API symbols when building a shared library (Windows uses dllexport; Unix uses visibility attribute)
    target_compile_definitions(beatsync_backend_shared PRIVATE BEATSYNC_CAPI_EXPORT)
    target_include_directories(beatsync_backend_shared PUBLIC ${CMAKE_SOURCE_DIR}/src)
    set_target_properties(beatsync_backend_shared PROPERTIES PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/src/backend/beatsync_capi.h")
    target_link_libraries(beatsync_backend_shared PRIVATE avformat avcodec avutil swresample swscale avfilter)

    # Ensure OPTIONAL_SOURCES (eg. TransitionLibrary) are explicitly added to shared target too
    target_sources(beatsync_backend_shared PRIVATE ${OPTIONAL_SOURCES})

    # Fail release builds if ENABLE_BEATNET_PYTHON is enabled (to avoid bundling Python in releases)
    if(ENABLE_BEATNET_PYTHON AND CMAKE_BUILD_TYPE MATCHES "Release")
        message(FATAL_ERROR "ENABLE_BEATNET_PYTHON must be OFF for Release builds. Set ENABLE_BEATNET_PYTHON=OFF to proceed.")
    endif()

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
        message(STATUS "ONNX Runtime found - attempting to enable native AI inference")
        if(EXISTS "${CMAKE_SOURCE_DIR}/src/audio/OnnxBeatDetector.cpp" AND EXISTS "${CMAKE_SOURCE_DIR}/src/audio/OnnxStemSeparator.cpp")
            list(APPEND OPTIONAL_SOURCES 
                src/audio/OnnxBeatDetector.cpp
                src/audio/OnnxStemSeparator.cpp
            )
            add_definitions(-DUSE_ONNX)
        else()
            message(WARNING "ONNX Runtime found, but ONNX source files are missing in src/audio/. ONNX inference will remain a stub until sources are added.")
        endif()
    else()
        # Provide actionable guidance to users when ONNX isn't found
        if(EXISTS "${CMAKE_SOURCE_DIR}/vcpkg.json")
            message(WARNING "ONNX Runtime not found. This repository includes a vcpkg manifest (vcpkg.json).")
            message(STATUS "Recommended steps:")
            message(STATUS "  1) Install vcpkg (https://github.com/microsoft/vcpkg).")
            message(STATUS "  2) From the repo root run: vcpkg bootstrap (Windows: .\\vcpkg\\bootstrap-vcpkg.bat).")
            message(STATUS "  3) Install packages: .\\vcpkg\\vcpkg.exe install --manifest")
            message(STATUS "  4) Reconfigure CMake with: -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake")
            message(STATUS "If you have a CUDA-enabled GPU and want GPU-accelerated ONNX, install 'onnxruntime-gpu' instead of 'onnxruntime'.")
            if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
                message(STATUS "Detected VCPKG_ROOT environment variable. You can reconfigure with -DCMAKE_TOOLCHAIN_FILE=$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
            endif()
        else()
            message(WARNING "ONNX Runtime not found. Install with vcpkg (e.g., 'vcpkg install onnxruntime' or 'vcpkg install onnxruntime-gpu') and reconfigure with the vcpkg toolchain file.")
        endif()
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

# Ensure optional sources are added to the backend library target
if(TARGET beatsync_backend_static)
    target_sources(beatsync_backend_static PRIVATE ${OPTIONAL_SOURCES})
endif()

if(TARGET beatsync_backend_shared)
    target_sources(beatsync_backend_shared PRIVATE ${OPTIONAL_SOURCES})
endif()

set(GUI_SOURCES
    src/GUI/MainWindow.cpp
    src/GUI/BeatVisualizer.cpp
    src/GUI/VideoPreview.cpp
    src/GUI/ProcessingThread.cpp
    src/GUI/SettingsManager.cpp
    src/GUI/PsychedelicTheme.cpp
    src/utils/LogArchiver.cpp
)

# CLI executable (existing)
add_executable(beatsync src/main.cpp)
# Link against the backend library and required FFmpeg libs
target_link_libraries(beatsync PRIVATE
    beatsync_backend_static
    avformat avcodec avutil swresample swscale avfilter
)
if(USE_ONNX AND onnxruntime_FOUND)
    target_link_libraries(beatsync PRIVATE onnxruntime::onnxruntime)
endif()

# GUI executable (NEW)
if(BUILD_GUI)
    if(WIN32)
        # Add icon resource only if the file exists. This avoids build failures when a placeholder
        # or no icon is present in the assets directory.
        if(EXISTS "${CMAKE_SOURCE_DIR}/assets/icon.rc")
            set(GUI_ICON_RC "${CMAKE_SOURCE_DIR}/assets/icon.rc")
            add_executable(TripSitter WIN32 src/GUI_main.cpp ${GUI_SOURCES} ${GUI_ICON_RC})
        else()
            add_executable(TripSitter WIN32 src/GUI_main.cpp ${GUI_SOURCES})
        endif()
    else()
        add_executable(TripSitter src/GUI_main.cpp ${GUI_SOURCES})
    endif()

    # Link GUI executable against backend and required libs
    target_link_libraries(TripSitter PRIVATE
        beatsync_backend_static
        avformat avcodec avutil swresample swscale avfilter
    )

    if(USE_ONNX AND onnxruntime_FOUND)
        target_link_libraries(TripSitter PRIVATE onnxruntime::onnxruntime)
    endif()

else()
    message(STATUS "GUI build is disabled (BUILD_GUI=OFF or wxWidgets not found)")
endif()

# Optional: link Essentia if requested and available
if(USE_ESSENTIA)
    find_package(Essentia CONFIG QUIET)
    if(Essentia_FOUND)
        message(STATUS "Essentia found - enabling Essentia analyzer support")
        target_link_libraries(beatsync PUBLIC Essentia::essent)
        if(BUILD_GUI)
            target_link_libraries(TripSitter PUBLIC Essentia::essent)
        endif()
        add_compile_definitions(HAVE_ESSENTIA)
    else()
        message(WARNING "USE_ESSENTIA enabled but Essentia package not found. EssentiaAnalyzer will remain a stub.")
    endif()
endif()

if(BUILD_GUI)

# macOS App Bundle configuration
if(APPLE)
    set_target_properties(TripSitter PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME "MTV Trip Sitter"
        MACOSX_BUNDLE_BUNDLE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.mtv.tripsitter"
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_SOURCE_DIR}/assets/Info.plist.in"
    )
endif()

# Output directory
set_target_properties(beatsync TripSitter PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)

# Copy assets to build directory
if(APPLE)
    # macOS: Copy assets into app bundle's MacOS folder (next to executable)
    add_custom_command(TARGET TripSitter POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        $<TARGET_BUNDLE_CONTENT_DIR:TripSitter>/MacOS/assets
    )

    # Optionally copy FFmpeg dylibs into the app bundle when FFMPEG_ROOT is provided
    if(FFMPEG_ROOT)
        file(GLOB FFMPEG_DYLIBS "${FFMPEG_ROOT}/lib/*.dylib" )
        if(FFMPEG_DYLIBS)
            add_custom_command(TARGET TripSitter POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_CONTENT_DIR:TripSitter>/MacOS/Frameworks"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FFMPEG_DYLIBS} "$<TARGET_BUNDLE_CONTENT_DIR:TripSitter>/MacOS/Frameworks/"
            )
            # Also install dylibs so cpack packages them
            install(FILES ${FFMPEG_DYLIBS} DESTINATION Frameworks OPTIONAL)
        endif()
    endif()
else()
    add_custom_command(TARGET TripSitter POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/assets
    )
endif()

# Ensure a specific wallpaper.png is copied and installed when present. This helps packaging
# and guarantees the file is present even if the assets directory is pruned by downstream tooling.
if(EXISTS "${CMAKE_SOURCE_DIR}/assets/wallpaper.png")
    if(APPLE)
        add_custom_command(TARGET TripSitter POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_CONTENT_DIR:TripSitter>/MacOS/assets"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/assets/wallpaper.png
            $<TARGET_BUNDLE_CONTENT_DIR:TripSitter>/MacOS/assets/wallpaper.png
        )
    else()
        add_custom_command(TARGET TripSitter POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/assets"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/assets/wallpaper.png
            ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/assets/wallpaper.png
        )
    endif()
    # Also ensure CPack includes the wallpaper explicitly
    install(FILES ${CMAKE_SOURCE_DIR}/assets/wallpaper.png DESTINATION . OPTIONAL)
endif()

endif()

# Backend smoke test executable (quick runtime check)
add_executable(backend_smoke src/tools/backend_smoke.cpp)
target_include_directories(backend_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(backend_smoke PRIVATE beatsync_backend_static avformat avcodec avutil swresample swscale avfilter)
set_target_properties(backend_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")

# Simple smoke test that loads the copied beatsync shared DLL and checks required exports
add_executable(dll_loader_smoke src/tools/dll_loader_smoke.cpp)
set_target_properties(dll_loader_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
if(WIN32)
    # No extra link libs required - uses Win32 LoadLibrary
else()
    # Link dl on platforms that require it (linux)
    target_link_libraries(dll_loader_smoke PRIVATE dl)
endif()

# Ensure CLI gets installed so CPack includes the beatsync executable
install(TARGETS beatsync RUNTIME DESTINATION .)

if(BUILD_GUI)
    # Copy FFmpeg DLLs to output directory (Windows only)
    if(WIN32)
        file(GLOB FFMPEG_DLLS "${FFMPEG_ROOT}/bin/*.dll")
        foreach(DLL ${FFMPEG_DLLS})
            add_custom_command(TARGET TripSitter POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL}
                $<TARGET_FILE_DIR:TripSitter>
            )
        endforeach()

        # Also install FFmpeg DLLs so they are included by CPack
        install(FILES ${FFMPEG_DLLS} DESTINATION .)
    endif()

    # Install TripSitter executable and assets for packaging
    if(APPLE)
        install(TARGETS TripSitter BUNDLE DESTINATION .)
    else()
        install(TARGETS TripSitter RUNTIME DESTINATION .)
    endif()
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/assets DESTINATION .)
endif()

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

# Packaging safety check: fail if Python is enabled in release builds
# This ensures packaged releases don't require Python
if(ENABLE_BEATNET_PYTHON)
    message(WARNING "ENABLE_BEATNET_PYTHON is ON - packaged builds will require Python!")
    message(WARNING "Set -DENABLE_BEATNET_PYTHON=OFF for release builds.")
    # Uncomment the following line to make this a hard error:
    # message(FATAL_ERROR "Cannot package with ENABLE_BEATNET_PYTHON=ON")
endif()

# CPack packaging configuration (ZIP + NSIS for Windows)
include(InstallRequiredSystemLibraries)
set(CPACK_PACKAGE_NAME "MTVTripSitter")
set(CPACK_PACKAGE_VENDOR "MTV")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "MTV Trip Sitter - Audio Beat Sync GUI")
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
    set(CPACK_NSIS_CONTACT "support@tripsitter.dev")
    set(CPACK_NSIS_HELP_LINK "https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI")
    set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI")

    # Icon for installer
    if(EXISTS "${CMAKE_SOURCE_DIR}/assets/icon.ico")
        set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/assets/icon.ico")
        set(CPACK_NSIS_MUI_UNIICON "${CMAKE_SOURCE_DIR}/assets/icon.ico")
        set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\TripSitter.exe")
    endif()

    # Start menu and desktop shortcuts
    set(CPACK_NSIS_CREATE_ICONS_EXTRA
        "CreateShortCut '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\MTV TripSitter.lnk' '$INSTDIR\\\\bin\\\\TripSitter.exe'"
    )
    set(CPACK_NSIS_DELETE_ICONS_EXTRA
        "Delete '$SMPROGRAMS\\\\$START_MENU\\\\MTV TripSitter.lnk'"
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
