cmake_minimum_required(VERSION 3.20)
project(TripSitterImGui LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(FetchContent)

# ---------------------------------------------------------------------------
# Dear ImGui (no upstream CMake; we compile the sources ourselves)
# ---------------------------------------------------------------------------
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.5
    GIT_SHALLOW    TRUE
)

# ---------------------------------------------------------------------------
# GLFW (windowing + input). Prefer system package, else fetch.
# ---------------------------------------------------------------------------
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4
        GIT_SHALLOW    TRUE
    )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
endif()

FetchContent_MakeAvailable(imgui)

# ImGui static library: core + GLFW/OpenGL3 backends
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw)

find_package(OpenGL REQUIRED)

# ---------------------------------------------------------------------------
# TripSitter ImGui application
# ---------------------------------------------------------------------------
add_executable(TripSitterImGui
    src/main.cpp
    src/TripSitterApp.cpp
    src/BackendLoader.cpp
    src/FileDialog.cpp
)

target_include_directories(TripSitterImGui PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/backend   # canonical beatsync_capi.h
)

target_link_libraries(TripSitterImGui PRIVATE imgui OpenGL::GL)

if(APPLE)
    target_compile_definitions(TripSitterImGui PRIVATE GL_SILENCE_DEPRECATION)
    target_link_libraries(TripSitterImGui PRIVATE
        "-framework Cocoa" "-framework IOKit" "-framework CoreVideo")
endif()

if(UNIX)
    target_link_libraries(TripSitterImGui PRIVATE ${CMAKE_DL_LIBS})
endif()

# Copy theme assets next to the binary so the app can find them at runtime.
add_custom_command(TARGET TripSitterImGui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:TripSitterImGui>/Resources
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/../Corpta.otf
        $<TARGET_FILE_DIR:TripSitterImGui>/Resources/Corpta.otf
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/../wallpaper.png
        $<TARGET_FILE_DIR:TripSitterImGui>/Resources/wallpaper.png
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/../TitleHeader.png
        $<TARGET_FILE_DIR:TripSitterImGui>/Resources/TitleHeader.png
    VERBATIM
)
