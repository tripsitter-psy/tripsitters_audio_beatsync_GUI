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

# Catch2-based tests (require BUILD_TESTS=ON to fetch Catch2)
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(catch2)

    # LogArchiver Catch2 stub test
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
    set_tests_properties(transition_library PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

    # Transition chaining tests
    add_executable(test_transition_chaining
        catch_main.cpp
        test_transition_chaining.cpp
    )

    target_include_directories(test_transition_chaining PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/backend)
    target_link_libraries(test_transition_chaining PRIVATE Catch2::Catch2 beatsync_backend_static avformat avcodec avutil swresample swscale avfilter)
    add_test(NAME transition_chaining COMMAND test_transition_chaining)
    set_tests_properties(transition_chaining PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

    # Backend API smoke test
    add_executable(test_backend_api
        catch_main.cpp
        test_backend_api.cpp
    )

    target_include_directories(test_backend_api PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/backend)
    target_link_libraries(test_backend_api PRIVATE Catch2::Catch2 beatsync_backend_static avformat avcodec avutil swresample swscale avfilter)
    add_test(NAME backend_api COMMAND test_backend_api)
    set_tests_properties(backend_api PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

    # ONNX AI tests (runs if ONNX Runtime available)
    if(USE_ONNX AND onnxruntime_FOUND)
        add_executable(test_onnx_ai
            catch_main.cpp
            test_onnx_ai.cpp
            test_ai_cleanup.cpp
        )

        target_include_directories(test_onnx_ai PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/backend)
        target_link_libraries(test_onnx_ai PRIVATE Catch2::Catch2 beatsync_backend_static avformat avcodec avutil swresample swscale avfilter onnxruntime::onnxruntime)

        # Allow CI or user to override test paths
        if(DEFINED TEST_AUDIO_FILE)
            target_compile_definitions(test_onnx_ai PRIVATE TEST_AUDIO_FILE_OVERRIDE="${TEST_AUDIO_FILE}")
        endif()
        if(DEFINED TEST_MODEL_DIR)
            target_compile_definitions(test_onnx_ai PRIVATE TEST_MODEL_DIR_OVERRIDE="${TEST_MODEL_DIR}")
        endif()

        add_test(NAME onnx_ai COMMAND test_onnx_ai)
        set_tests_properties(onnx_ai PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        # Label this test for CUDA so CI can select it via ctest -L cuda
        set_tests_properties(onnx_ai PROPERTIES LABELS "cuda")
    endif()

    # AudioFlux test (if AudioFlux is available)
    # Note: test_audioflux.cpp is a standalone test with its own main(), not a Catch2 test
    if(USE_AUDIOFLUX)
        add_executable(test_audioflux
            test_audioflux.cpp
        )
        target_include_directories(test_audioflux PRIVATE ${CMAKE_SOURCE_DIR}/src ${AUDIOFLUX_ROOT}/include)
        target_link_libraries(test_audioflux PRIVATE beatsync_backend_static avformat avcodec avutil swresample swscale avfilter)
        add_test(NAME audioflux COMMAND test_audioflux)
        set_tests_properties(audioflux PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endif()
endif()

# GUI wallpaper tests removed - wxWidgets GUI has been deprecated and deleted
# The GUI is now implemented in Unreal Engine (TripSitter standalone app)
    # Spectral flux tests
    add_executable(test_spectral_flux
        catch_main.cpp
        test_spectral_flux.cpp
        ../src/audio/SpectralFlux.cpp
    )

    target_include_directories(test_spectral_flux PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_spectral_flux PRIVATE Catch2::Catch2)
    add_test(NAME spectral_flux COMMAND test_spectral_flux)
    set_tests_properties(spectral_flux PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
