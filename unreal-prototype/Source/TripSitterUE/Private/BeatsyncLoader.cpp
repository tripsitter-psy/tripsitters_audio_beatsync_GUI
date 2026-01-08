#include "BeatsyncLoader.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "Misc/OutputDeviceNull.h"
#include <cassert>

// Function pointer types
using bs_resolve_ffmpeg_path_t = const char* (*)();
using bs_create_audio_analyzer_t = void* (*)();
using bs_destroy_audio_analyzer_t = void (*)(void*);

struct FBeatsyncApi
{
    void* DllHandle = nullptr;
    bs_resolve_ffmpeg_path_t resolve_ffmpeg = nullptr;
    bs_create_audio_analyzer_t create_analyzer = nullptr;
    bs_destroy_audio_analyzer_t destroy_analyzer = nullptr;
};

static FBeatsyncApi GApi;

bool FBeatsyncLoader::Initialize()
{
    if (GApi.DllHandle) return true;

    // Expected locations (platform-aware):
    // Windows: <repo>/unreal-prototype/ThirdParty/beatsync/lib/x64/beatsync_backend.dll
    // macOS:   <repo>/unreal-prototype/ThirdParty/beatsync/lib/Mac/libbeatsync_backend.dylib
    // Linux:   <repo>/unreal-prototype/ThirdParty/beatsync/lib/Linux/libbeatsync_backend.so

    FString ModuleDir = FPaths::ConvertRelativePathToFull(FPaths::ProjectDir()); // ProjectDir points to UE project when running in editor; for prototype use ModuleDirectory fallback

    FString Filename;
    FString Subdir;
#if PLATFORM_WINDOWS
    Filename = TEXT("beatsync_backend.dll");
    Subdir = TEXT("x64");
#elif PLATFORM_MAC
    Filename = TEXT("libbeatsync_backend.dylib");
    Subdir = TEXT("Mac");
#else
    Filename = TEXT("libbeatsync_backend.so");
    Subdir = TEXT("Linux");
#endif

    FString DllPath = FPaths::Combine(FPaths::ProjectDir(), TEXT(".."), TEXT("unreal-prototype"), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("lib"), Subdir, Filename);

    // If ProjectDir is empty in some contexts, try module relative path
    if (!FPaths::FileExists(DllPath)) {
        FString Relative = FPaths::Combine(FPaths::ConvertRelativePathToFull(FPaths::EngineDir()), TEXT(".."));
        DllPath = FPaths::Combine(Relative, TEXT("unreal-prototype"), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("lib"), Subdir, Filename);
    }

    if (!FPaths::FileExists(DllPath)) {
        UE_LOG(LogTemp, Error, TEXT("Beatsync library not found at: %s"), *DllPath);
        return false;
    }

    GApi.DllHandle = FPlatformProcess::GetDllHandle(*DllPath);
    if (!GApi.DllHandle) {
        UE_LOG(LogTemp, Error, TEXT("Failed to load Beatsync library: %s"), *DllPath);
        return false;
    }

    GApi.resolve_ffmpeg = (bs_resolve_ffmpeg_path_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_resolve_ffmpeg_path"));
    GApi.create_analyzer = (bs_create_audio_analyzer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_create_audio_analyzer"));
    GApi.destroy_analyzer = (bs_destroy_audio_analyzer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_destroy_audio_analyzer"));

    if (!GApi.resolve_ffmpeg || !GApi.create_analyzer || !GApi.destroy_analyzer) {
        UE_LOG(LogTemp, Error, TEXT("Required symbols not found in Beatsync DLL"));
        FPlatformProcess::FreeDllHandle(GApi.DllHandle);
        GApi.DllHandle = nullptr;
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("Beatsync DLL loaded: %s"), *DllPath);
    return true;
}

void FBeatsyncLoader::Shutdown()
{
    if (GApi.DllHandle) {
        FPlatformProcess::FreeDllHandle(GApi.DllHandle);
        GApi.DllHandle = nullptr;
        GApi.resolve_ffmpeg = nullptr;
        GApi.create_analyzer = nullptr;
        GApi.destroy_analyzer = nullptr;
    }
}

FString FBeatsyncLoader::ResolveFFmpegPath()
{
    if (!GApi.resolve_ffmpeg) return FString();
    const char* p = GApi.resolve_ffmpeg();
    return p ? FString(p) : FString();
}

void* FBeatsyncLoader::CreateAnalyzer()
{
    if (!GApi.create_analyzer) return nullptr;
    return GApi.create_analyzer();
}

void FBeatsyncLoader::DestroyAnalyzer(void* handle)
{
    if (!GApi.destroy_analyzer) return;
    GApi.destroy_analyzer(handle);
}
