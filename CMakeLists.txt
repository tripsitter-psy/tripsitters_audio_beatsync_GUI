cmake_minimum_required(VERSION 3.15)
project(TripSitterBeatSync)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific FFmpeg setup
if(WIN32)
    set(FFMPEG_ROOT "C:/ffmpeg-dev/ffmpeg-master-latest-win64-gpl-shared")
    include_directories(${FFMPEG_ROOT}/include)
    link_directories(${FFMPEG_ROOT}/lib)
elseif(APPLE)
    # Use pkg-config to find FFmpeg on macOS (installed via Homebrew)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
        libavformat
        libavcodec
        libavutil
        libswresample
        libswscale
        libavfilter
    )
endif()

# Platform-specific wxWidgets setup
if(WIN32)
    set(wxWidgets_ROOT_DIR "C:/wxWidgets-3.2.4")
    set(wxWidgets_CONFIGURATION mswu)
endif()
find_package(wxWidgets REQUIRED COMPONENTS core base adv gl)
include(${wxWidgets_USE_FILE})

# Source files
set(CORE_SOURCES
    src/audio/AudioAnalyzer.cpp
    src/audio/BeatGrid.cpp
    src/video/VideoProcessor.cpp
    src/video/VideoWriter.cpp
)

set(GUI_SOURCES
    src/gui/MainWindow.cpp
    src/gui/BeatVisualizer.cpp
    src/gui/VideoPreview.cpp
    src/gui/ProcessingThread.cpp
    src/gui/SettingsManager.cpp
)

# CLI executable (existing)
add_executable(beatsync src/main.cpp ${CORE_SOURCES})
if(APPLE)
    target_link_libraries(beatsync PkgConfig::FFMPEG)
else()
    target_link_libraries(beatsync avformat avcodec avutil swresample swscale avfilter)
endif()

# GUI executable (NEW)
if(WIN32)
    # Add icon resource only if the file exists. This avoids build failures when a placeholder
    # or no icon is present in the assets directory.
    if(EXISTS "${CMAKE_SOURCE_DIR}/assets/icon.rc")
        set(GUI_ICON_RC "${CMAKE_SOURCE_DIR}/assets/icon.rc")
        add_executable(TripSitter WIN32 src/gui_main.cpp ${CORE_SOURCES} ${GUI_SOURCES} ${GUI_ICON_RC})
    else()
        add_executable(TripSitter WIN32 src/gui_main.cpp ${CORE_SOURCES} ${GUI_SOURCES})
    endif()
else()
    add_executable(TripSitter src/gui_main.cpp ${CORE_SOURCES} ${GUI_SOURCES})
endif()

if(APPLE)
    target_link_libraries(TripSitter
        ${wxWidgets_LIBRARIES}
        PkgConfig::FFMPEG
    )
else()
    target_link_libraries(TripSitter
        ${wxWidgets_LIBRARIES}
        avformat avcodec avutil swresample swscale avfilter
    )
endif()

# Output directory
set_target_properties(beatsync TripSitter PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)

# Copy assets to build directory
add_custom_command(TARGET TripSitter POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets
    ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/assets
)

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
install(TARGETS TripSitter RUNTIME DESTINATION .)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/assets DESTINATION .)

# CPack packaging configuration (ZIP + NSIS for Windows)
include(InstallRequiredSystemLibraries)
set(CPACK_PACKAGE_NAME "TripSitter")
set(CPACK_PACKAGE_VENDOR "TripSitter")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "TripSitter - Audio Beat Sync GUI")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "TripSitter")
    set(CPACK_NSIS_CONTACT "support@example.com")
elseif(APPLE)
    set(CPACK_GENERATOR "ZIP;DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "TripSitter")
    set(CPACK_BUNDLE_NAME "TripSitter")
else()
    set(CPACK_GENERATOR "ZIP;TGZ")
endif()

include(CPack)
