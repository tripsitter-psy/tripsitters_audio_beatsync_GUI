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

# Catch2-based stub test (disabled until BUILD_TESTS is ON). This provides a minimal
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

# Transition library tests
add_executable(test_transition_library
    catch_main.cpp
    test_transition_library.cpp
    ../src/video/TransitionLibrary.cpp
)

target_include_directories(test_transition_library PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_transition_library PRIVATE Catch2::Catch2)
add_test(NAME transition_library COMMAND test_transition_library)

# Header paint integration test
add_executable(test_header_paint
    catch_main.cpp
    test_header_paint.cpp
)

target_include_directories(test_header_paint PRIVATE ${CMAKE_SOURCE_DIR}/src)
# Link against wxWidgets so we can create windows and run a minimal paint
target_link_libraries(test_header_paint PRIVATE Catch2::Catch2 ${wxWidgets_LIBRARIES})
# Run the test under Xvfb in environments where a display is not available
add_test(NAME header_paint COMMAND xvfb-run -s "-screen 0 1280x800x24" $<TARGET_FILE:test_header_paint>)

# Transition chaining tests
add_executable(test_transition_chaining
    catch_main.cpp
    test_transition_chaining.cpp
    ../src/video/TransitionLibrary.cpp
    ../src/video/VideoWriter.cpp
    ../src/video/VideoProcessor.cpp
    ../src/audio/BeatGrid.cpp
    ../src/audio/AudioAnalyzer.cpp
    ../src/audio/BeatNetBridge.cpp
    ../src/audio/StemSeparator.cpp
    ../src/tracing/Tracing.cpp
    ../src/utils/ProcessUtils.cpp
)

target_include_directories(test_transition_chaining PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_transition_chaining PRIVATE Catch2::Catch2 avformat avcodec avutil swresample swscale avfilter)
add_test(NAME transition_chaining COMMAND test_transition_chaining)

# Audio transition filter tests
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_audio_transition_filter.cpp")
    add_executable(test_audio_transition_filter
        catch_main.cpp
        test_audio_transition_filter.cpp
        ../src/video/VideoWriter.cpp
        ../src/video/VideoProcessor.cpp
        ../src/video/TransitionLibrary.cpp
        ../src/audio/BeatGrid.cpp
        ../src/audio/AudioAnalyzer.cpp
        ../src/audio/BeatNetBridge.cpp
        ../src/audio/StemSeparator.cpp
    )

    target_include_directories(test_audio_transition_filter PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_audio_transition_filter PRIVATE Catch2::Catch2 avformat avcodec avutil swresample swscale avfilter)
    add_test(NAME audio_transition_filter COMMAND test_audio_transition_filter)
else()
    message(STATUS "Skipping test_audio_transition_filter - source not present")
endif()
