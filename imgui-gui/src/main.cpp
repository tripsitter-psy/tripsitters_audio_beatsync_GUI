// TripSitter ImGui frontend - application entry point.
//
// Sets up a GLFW + OpenGL3 window, loads the neon theme and Corpta font, loads
// the beatsync backend, then runs the ImGui frame loop driving TripSitterApp.
#include "TripSitterApp.h"
#include "Theme.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <string>

static void GlfwErrorCallback(int error, const char* description)
{
    std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

int main(int, char**)
{
    glfwSetErrorCallback(GlfwErrorCallback);
    if (!glfwInit())
        return 1;

    // OpenGL 3.2 core (works on macOS, Linux, Windows).
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 820, "TripSitter", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    Theme::Apply();

    // Load the Corpta display font (copied next to the binary by CMake); fall
    // back to the built-in font if it is missing.
    io.Fonts->AddFontDefault();
    const char* fontPath = "Resources/Corpta.otf";
    if (FILE* f = std::fopen(fontPath, "rb")) { std::fclose(f);
        io.Fonts->AddFontFromFileTTF(fontPath, 18.0f);
    }

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load the audio/video backend (degrades gracefully if not found).
    BackendLoader backend;
    if (backend.Load())
    {
        if (backend.bs_init) backend.bs_init();
        std::printf("Backend loaded: %s (%s)\n",
                    backend.LoadedPath().c_str(),
                    backend.bs_get_version ? backend.bs_get_version() : "?");
    }
    else
    {
        std::fprintf(stderr, "Backend not loaded: %s\n", backend.LastError().c_str());
    }

    TripSitterApp app(backend);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        app.Draw();

        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(Theme::DarkBg.x, Theme::DarkBg.y, Theme::DarkBg.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    if (backend.IsLoaded() && backend.bs_shutdown) backend.bs_shutdown();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
