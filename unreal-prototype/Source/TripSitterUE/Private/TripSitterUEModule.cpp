#include "TripSitterUEModule.h"
#include "Modules/ModuleManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "Misc/OutputDeviceNull.h"

IMPLEMENT_MODULE(FTripSitterUEModule, TripSitterUE)

namespace {
using bs_init_tracing_t = int (*)(const char*);
using bs_shutdown_tracing_t = void (*)();

static void* BeatsyncDllHandle = nullptr;

static FString GetBeatsyncDllPath()
{
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

    if (!FPaths::FileExists(DllPath)) {
        // Try engine-relative fallback
        FString Relative = FPaths::Combine(FPaths::ConvertRelativePathToFull(FPaths::EngineDir()), TEXT(".."));
        DllPath = FPaths::Combine(Relative, TEXT("unreal-prototype"), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("lib"), Subdir, Filename);
    }

    return DllPath;
}

static void CallBackendInitTracing(const FString& ServiceName)
{
    // Attempt to locate the backend shared library (same logic as BeatsyncLoader)
    FString DllPath = GetBeatsyncDllPath();

    BeatsyncDllHandle = FPlatformProcess::GetDllHandle(*DllPath);
    if (!BeatsyncDllHandle) {
        UE_LOG(LogTemp, Warning, TEXT("TripSitterUEModule: Unable to find beatsync backend DLL at %s"), *DllPath);
        return;
    }

    void* Symbol = FPlatformProcess::GetDllExport(BeatsyncDllHandle, TEXT("bs_initialize_tracing"));
    if (!Symbol) {
        UE_LOG(LogTemp, Warning, TEXT("TripSitterUEModule: bs_initialize_tracing not found in backend"));
        return;
    }

    bs_init_tracing_t Init = reinterpret_cast<bs_init_tracing_t>(Symbol);
    FString UTF8 = ServiceName;
    Init(TCHAR_TO_ANSI(*UTF8));
}

static void CallBackendShutdownTracing()
{
    if (!BeatsyncDllHandle) return;

    void* Symbol = FPlatformProcess::GetDllExport(BeatsyncDllHandle, TEXT("bs_shutdown_tracing"));
    if (!Symbol) return;

    bs_shutdown_tracing_t Shutdown = reinterpret_cast<bs_shutdown_tracing_t>(Symbol);
    Shutdown();
}
} // end anonymous namespace

void FTripSitterUEModule::StartupModule()
{
    // Try to init tracing with service name "tripsitter"
    CallBackendInitTracing(TEXT("tripsitter"));
}

void FTripSitterUEModule::ShutdownModule()
{
    // Try to shutdown tracing
    CallBackendShutdownTracing();

    // Free the DLL handle
    if (BeatsyncDllHandle) {
        FPlatformProcess::FreeDllHandle(BeatsyncDllHandle);
        BeatsyncDllHandle = nullptr;
    }
}
