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

# wxWidgets setup
# On Windows we ship a default path; on macOS/Linux we let find_package find it
# Set USE_WXUNIVERSAL=ON to use wxUniversal build with custom theming
option(USE_WXUNIVERSAL "Use wxUniversal build for custom theming" OFF)

# Optional AI / advanced feature flags (stubs included; wire deps when available)
option(USE_ESSENTIA "Enable Essentia-based beat analyzer" OFF)
option(USE_BEATNET "Enable BeatNet Python bridge" OFF)
option(USE_DEMUCS "Enable Demucs stem separation" OFF)
option(USE_GLTRANSITIONS "Enable GLSL transitions library" OFF)
option(USE_AUDIOREACTIVE "Enable FFT-driven audio-reactive effects" OFF)

if(WIN32)
    set(wxWidgets_ROOT_DIR "C:/wxWidgets-3.2.4")
    if(USE_WXUNIVERSAL)
        set(wxWidgets_CONFIGURATION mswunivu)
        set(wxWidgets_LIB_DIR "C:/wxWidgets-3.2.4/lib/vc_x64_dll")
        add_definitions(-D__WXUNIVERSAL__)
    else()
        set(wxWidgets_CONFIGURATION mswu)
    endif()
endif()
find_package(wxWidgets REQUIRED COMPONENTS core base adv)
include(${wxWidgets_USE_FILE})

# Source files
set(CORE_SOURCES
    src/audio/AudioAnalyzer.cpp
    src/audio/BeatGrid.cpp
    src/video/VideoProcessor.cpp
    src/video/VideoWriter.cpp
)

# Optional feature sources (always compiled as stubs; actual libs wired via flags)
set(OPTIONAL_SOURCES)
if(USE_ESSENTIA)
    list(APPEND OPTIONAL_SOURCES src/audio/EssentiaAnalyzer.cpp)
    add_definitions(-DUSE_ESSENTIA)
endif()
if(USE_BEATNET)
    list(APPEND OPTIONAL_SOURCES src/audio/BeatNetBridge.cpp)
    add_definitions(-DUSE_BEATNET)
endif()
if(USE_DEMUCS)
    list(APPEND OPTIONAL_SOURCES src/audio/StemSeparator.cpp)
    add_definitions(-DUSE_DEMUCS)
endif()
if(USE_GLTRANSITIONS)
    list(APPEND OPTIONAL_SOURCES src/video/TransitionLibrary.cpp)
    add_definitions(-DUSE_GLTRANSITIONS)
endif()
if(USE_AUDIOREACTIVE)
    list(APPEND OPTIONAL_SOURCES src/audio/FFTAnalyzer.cpp src/video/AudioReactiveEffects.cpp)
    add_definitions(-DUSE_AUDIOREACTIVE)
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
add_executable(beatsync src/main.cpp ${CORE_SOURCES} ${OPTIONAL_SOURCES})
target_link_libraries(beatsync
    avformat avcodec avutil swresample swscale avfilter
)

# GUI executable (NEW)
if(WIN32)
    # Add icon resource only if the file exists. This avoids build failures when a placeholder
    # or no icon is present in the assets directory.
    if(EXISTS "${CMAKE_SOURCE_DIR}/assets/icon.rc")
        set(GUI_ICON_RC "${CMAKE_SOURCE_DIR}/assets/icon.rc")
        add_executable(TripSitter WIN32 src/GUI_main.cpp ${CORE_SOURCES} ${GUI_SOURCES} ${OPTIONAL_SOURCES} ${GUI_ICON_RC})
    else()
        add_executable(TripSitter WIN32 src/GUI_main.cpp ${CORE_SOURCES} ${GUI_SOURCES} ${OPTIONAL_SOURCES})
    endif()
else()
    add_executable(TripSitter src/GUI_main.cpp ${CORE_SOURCES} ${GUI_SOURCES} ${OPTIONAL_SOURCES})
endif()

target_link_libraries(TripSitter
    ${wxWidgets_LIBRARIES}
    avformat avcodec avutil swresample swscale avfilter
)

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
else()
    add_custom_command(TARGET TripSitter POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/assets
    )
endif()

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

# Platform-specific CPack generators
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "MTV Trip Sitter")
    set(CPACK_NSIS_CONTACT "support@example.com")
elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "MTV Trip Sitter")
    set(CPACK_DMG_FORMAT "UDZO")
else()
    set(CPACK_GENERATOR "TGZ")
endif()

# Tests (disabled by default). Enable with -DBUILD_TESTS=ON
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

include(CPack)
