# Tests for BeatSyncEditor

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

# FFTAnalyzer test skeleton
add_executable(test_fft_analyzer
    catch_main.cpp
    test_fft_analyzer.cpp
)

target_include_directories(test_fft_analyzer PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_fft_analyzer PRIVATE Catch2::Catch2)
add_test(NAME fft_analyzer COMMAND test_fft_analyzer)
set_tests_properties(fft_analyzer PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# BeatNetBridge test skeleton
add_executable(test_beatnet_bridge
    catch_main.cpp
    test_beatnet_bridge.cpp
)

target_include_directories(test_beatnet_bridge PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_beatnet_bridge PRIVATE Catch2::Catch2)
add_test(NAME beatnet_bridge COMMAND test_beatnet_bridge)
set_tests_properties(beatnet_bridge PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

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
