cmake_minimum_required(VERSION 3.15)
project(TripSitterBeatSync)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific FFmpeg setup
if(WIN32)
    if(NOT DEFINED FFMPEG_ROOT)
        set(FFMPEG_ROOT "C:/ffmpeg-dev/ffmpeg-master-latest-win64-gpl-shared")
    endif()
    include_directories(${FFMPEG_ROOT}/include)
    link_directories(${FFMPEG_ROOT}/lib)
elseif(APPLE)
    # On macOS, support both explicit FFMPEG_ROOT and pkg-config (Homebrew)
    if(DEFINED FFMPEG_ROOT)
        # Use explicit FFMPEG_ROOT if provided (e.g., from vcpkg or brew --prefix)
        include_directories(${FFMPEG_ROOT}/include)
        link_directories(${FFMPEG_ROOT}/lib)
        # For vcpkg, libraries might be in different locations
        find_library(AVFORMAT_LIB avformat HINTS ${FFMPEG_ROOT}/lib)
        find_library(AVCODEC_LIB avcodec HINTS ${FFMPEG_ROOT}/lib)
        find_library(AVUTIL_LIB avutil HINTS ${FFMPEG_ROOT}/lib)
        find_library(SWRESAMPLE_LIB swresample HINTS ${FFMPEG_ROOT}/lib)
        find_library(SWSCALE_LIB swscale HINTS ${FFMPEG_ROOT}/lib)
        find_library(AVFILTER_LIB avfilter HINTS ${FFMPEG_ROOT}/lib)
    else()
        # Use pkg-config to find FFmpeg (installed via Homebrew)
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
endif()

# wxWidgets setup
# Set USE_WXUNIVERSAL=ON to use wxUniversal build with custom theming
option(USE_WXUNIVERSAL "Use wxUniversal build for custom theming" OFF)

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
add_executable(beatsync src/main.cpp ${CORE_SOURCES})
if(APPLE)
    if(DEFINED FFMPEG_ROOT)
        # When using explicit FFMPEG_ROOT, link libraries directly
        target_link_libraries(beatsync 
            ${AVFORMAT_LIB} ${AVCODEC_LIB} ${AVUTIL_LIB} 
            ${SWRESAMPLE_LIB} ${SWSCALE_LIB} ${AVFILTER_LIB})
    else()
        # When using pkg-config, use the imported target
        target_link_libraries(beatsync PkgConfig::FFMPEG)
    endif()
else()
    target_link_libraries(beatsync avformat avcodec avutil swresample swscale avfilter)
endif()

# GUI executable (NEW)
if(WIN32)
    # Add icon resource only if the file exists. This avoids build failures when a placeholder
    # or no icon is present in the assets directory.
    if(EXISTS "${CMAKE_SOURCE_DIR}/assets/icon.rc")
        set(GUI_ICON_RC "${CMAKE_SOURCE_DIR}/assets/icon.rc")
        add_executable(TripSitter WIN32 src/GUI_main.cpp ${CORE_SOURCES} ${GUI_SOURCES} ${GUI_ICON_RC})
    else()
        add_executable(TripSitter WIN32 src/GUI_main.cpp ${CORE_SOURCES} ${GUI_SOURCES})
    endif()
else()
    add_executable(TripSitter src/GUI_main.cpp ${CORE_SOURCES} ${GUI_SOURCES})
endif()

if(APPLE)
    if(DEFINED FFMPEG_ROOT)
        # When using explicit FFMPEG_ROOT, link libraries directly
        target_link_libraries(TripSitter
            ${wxWidgets_LIBRARIES}
            ${AVFORMAT_LIB} ${AVCODEC_LIB} ${AVUTIL_LIB} 
            ${SWRESAMPLE_LIB} ${SWSCALE_LIB} ${AVFILTER_LIB}
        )
    else()
        # When using pkg-config, use the imported target
        target_link_libraries(TripSitter
            ${wxWidgets_LIBRARIES}
            PkgConfig::FFMPEG
        )
    endif()
else()
    target_link_libraries(TripSitter
        ${wxWidgets_LIBRARIES}
        avformat avcodec avutil swresample swscale avfilter
    )
endif()

# macOS App Bundle configuration
if(APPLE)
    set_target_properties(TripSitter PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME "TripSitter"
        MACOSX_BUNDLE_BUNDLE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.tripsitter.beatsync"
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_SOURCE_DIR}/assets/Info.plist.in"
    )
endif()

# Output directory
set_target_properties(beatsync TripSitter PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)

# Copy assets to build directory
if(APPLE)
    # For macOS app bundle, copy to Resources folder
    add_custom_command(TARGET TripSitter POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        $<TARGET_BUNDLE_CONTENT_DIR:TripSitter>/Resources/assets
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

# CPack packaging configuration
include(InstallRequiredSystemLibraries)
set(CPACK_PACKAGE_NAME "TripSitter")
set(CPACK_PACKAGE_VENDOR "TripSitter")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "TripSitter - Audio Beat Sync GUI")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")

# Platform-specific CPack generators
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "TripSitter")
    set(CPACK_NSIS_CONTACT "support@example.com")
elseif(APPLE)
    set(CPACK_GENERATOR "ZIP;DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "TripSitter")
    set(CPACK_DMG_FORMAT "UDZO")
    set(CPACK_BUNDLE_NAME "TripSitter")
else()
    set(CPACK_GENERATOR "ZIP;TGZ")
endif()

# Tests (disabled by default). Enable with -DBUILD_TESTS=ON
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

include(CPack)
