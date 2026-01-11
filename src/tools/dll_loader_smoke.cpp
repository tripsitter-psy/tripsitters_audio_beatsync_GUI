/**
 * DLL Loader Smoke Test
 * Verifies the shared library exports can be loaded dynamically
 * Used in CI and by Unreal plugin loader validation
 */

#include <iostream>
#include <cstdlib>

#if defined(_WIN32)
    #include <windows.h>
    #define LOAD_LIBRARY(path) LoadLibraryA(path)
    #define GET_PROC(lib, name) GetProcAddress((HMODULE)lib, name)
    #define FREE_LIBRARY(lib) FreeLibrary((HMODULE)lib)
    #define LIB_HANDLE HMODULE
#else
    #include <dlfcn.h>
    #define LOAD_LIBRARY(path) dlopen(path, RTLD_NOW)
    #define GET_PROC(lib, name) dlsym(lib, name)
    #define FREE_LIBRARY(lib) dlclose(lib)
    #define LIB_HANDLE void*
#endif

// Expected C API exports
typedef void* (*bs_create_audio_analyzer_fn)();
typedef void (*bs_destroy_audio_analyzer_fn)(void*);
typedef const char* (*bs_resolve_ffmpeg_path_fn)();

int main(int argc, char* argv[]) {
    std::cout << "=== DLL Loader Smoke Test ===\n";

    // Determine library path
    const char* libPath = nullptr;
    if (argc > 1) {
        libPath = argv[1];
    } else {
#if defined(_WIN32)
        // Default paths to check
        const char* defaultPaths[] = {
            "beatsync_backend_shared.dll",
            "../unreal-prototype/ThirdParty/beatsync/lib/x64/beatsync_backend_shared.dll",
            "bin/Release/beatsync_backend_shared.dll",
            nullptr
        };
#else
        const char* defaultPaths[] = {
            "libbeatsync_backend_shared.so",
            "libbeatsync_backend_shared.dylib",
            "../unreal-prototype/ThirdParty/beatsync/lib/Mac/libbeatsync_backend.dylib",
            nullptr
        };
#endif
        for (int i = 0; defaultPaths[i]; ++i) {
            LIB_HANDLE test = LOAD_LIBRARY(defaultPaths[i]);
            if (test) {
                FREE_LIBRARY(test);
                libPath = defaultPaths[i];
                break;
            }
        }
    }

    if (!libPath) {
        std::cerr << "ERROR: No library path specified and default paths not found.\n";
        std::cerr << "Usage: " << argv[0] << " <path-to-dll>\n";
        return 1;
    }

    std::cout << "[1/4] Loading library: " << libPath << "... ";
    LIB_HANDLE lib = LOAD_LIBRARY(libPath);
    if (!lib) {
#if defined(_WIN32)
        DWORD err = GetLastError();
        std::cerr << "FAILED (error code: " << err << ")\n";
#else
        std::cerr << "FAILED: " << dlerror() << "\n";
#endif
        return 1;
    }
    std::cout << "OK\n";

    // Check exports
    int errors = 0;

    std::cout << "[2/4] Looking up bs_create_audio_analyzer... ";
    auto create_analyzer = (bs_create_audio_analyzer_fn)GET_PROC(lib, "bs_create_audio_analyzer");
    if (create_analyzer) {
        std::cout << "OK\n";
    } else {
        std::cout << "FAILED\n";
        errors++;
    }

    std::cout << "[3/4] Looking up bs_destroy_audio_analyzer... ";
    auto destroy_analyzer = (bs_destroy_audio_analyzer_fn)GET_PROC(lib, "bs_destroy_audio_analyzer");
    if (destroy_analyzer) {
        std::cout << "OK\n";
    } else {
        std::cout << "FAILED\n";
        errors++;
    }

    std::cout << "[4/4] Looking up bs_resolve_ffmpeg_path... ";
    auto resolve_ffmpeg = (bs_resolve_ffmpeg_path_fn)GET_PROC(lib, "bs_resolve_ffmpeg_path");
    if (resolve_ffmpeg) {
        std::cout << "OK\n";
    } else {
        std::cout << "FAILED\n";
        errors++;
    }

    // Optional: Try calling the functions
    if (create_analyzer && destroy_analyzer) {
        std::cout << "\n[BONUS] Testing create/destroy cycle... ";
        void* analyzer = create_analyzer();
        if (analyzer) {
            destroy_analyzer(analyzer);
            std::cout << "OK\n";
        } else {
            std::cout << "create returned null (may be OK if deps missing)\n";
        }
    }

    if (resolve_ffmpeg) {
        std::cout << "[BONUS] Testing FFmpeg path resolution... ";
        const char* path = resolve_ffmpeg();
        if (path && path[0]) {
            std::cout << "OK: " << path << "\n";
        } else {
            std::cout << "(empty - FFmpeg may not be installed)\n";
        }
    }

    FREE_LIBRARY(lib);

    if (errors > 0) {
        std::cerr << "\n=== " << errors << " export(s) missing ===\n";
        return 1;
    }

    std::cout << "\n=== All DLL exports verified ===\n";
    return 0;
}
